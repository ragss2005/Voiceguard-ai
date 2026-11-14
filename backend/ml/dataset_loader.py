import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from backend.config import DATA_ROOT, CACHE_DIR, FEATURE_DIM
from backend.ml.features import extract_features, AudioValidationError

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt"
}

def find_protocol_file(root: Path, split: str) -> Optional[Path]:
    """Finds the protocol file for a specific partition under the dataset root."""
    target_name = PROTOCOLS.get(split)
    if not target_name:
        return None
    
    # 1. Direct search in root or root/ASVspoof2019_LA_protocols/
    candidates = list(root.rglob(target_name))
    if candidates:
        return candidates[0]
    return None

def build_audio_index(root: Path) -> Dict[str, Path]:
    """
    Builds a dictionary mapping audio filenames to their full Path.
    Indexes once to avoid recursive tree walking per sample.
    Supports .flac, .wav, .mp3, .ogg.
    """
    audio_index = {}
    for ext in ("*.flac", "*.wav", "*.mp3", "*.ogg"):
        for path in root.rglob(ext):
            audio_index[path.name] = path
            # Also store without extension in case protocol file omits it
            audio_index[path.stem] = path
    return audio_index

def parse_protocol(protocol_path: Path, limit: Optional[int] = None) -> List[Tuple[str, int, str]]:
    """
    Parses an ASVspoof 2019 LA protocol file.
    Protocol format: [SPEAKER_ID] [AUDIO_FILE_NAME] [ENVIRONMENT_ID] [ATTACK_ID] [KEY]
    where KEY is 'bonafide' or 'spoof'.
    
    Returns:
        List of tuples: (filename_with_ext, label_int, attack_type)
        bonafide = 0, spoof = 1
    """
    rows = []
    lines = protocol_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            audio_id = parts[1]
            attack_id = parts[3]
            key = parts[4].lower()
            if key in ("bonafide", "spoof"):
                label = 0 if key == "bonafide" else 1
                filename = f"{audio_id}.flac"
                rows.append((filename, label, attack_id))
                if limit and len(rows) >= limit:
                    break
    return rows

def get_cache_path(split: str, limit: Optional[int] = None) -> Path:
    """Returns path to cached .npz feature file."""
    tag = f"_{limit}" if limit else ""
    return CACHE_DIR / f"features_{split}{tag}.npz"

def load_cached_features(cache_file: Path) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Loads extracted features from cache if available and valid."""
    if cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=True)
            X = data["X"]
            y = data["y"]
            filenames = data["filenames"].tolist() if "filenames" in data else []
            if X.ndim == 2 and X.shape[1] == FEATURE_DIM and len(X) == len(y):
                return X, y, filenames
        except Exception:
            pass
    return None

def save_cached_features(cache_file: Path, X: np.ndarray, y: np.ndarray, filenames: List[str]):
    """Saves extracted features and labels to .npz cache."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y, filenames=np.asarray(filenames))

def load_partition(
    root: Optional[Path] = None,
    split: str = "train",
    limit: Optional[int] = None,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads an official ASVspoof partition (train, dev, or eval).
    Extracts features if not cached.
    
    Returns:
        X (np.ndarray of shape (N, 80)): extracted feature matrix
        y (np.ndarray of shape (N,)): labels (0 = bonafide, 1 = spoof)
        stats (dict): extraction statistics (missing, corrupt, bonafide, spoof counts)
    """
    if root is None:
        root = DATA_ROOT
    root = Path(root)

    cache_file = get_cache_path(split, limit)
    if use_cache:
        cached = load_cached_features(cache_file)
        if cached is not None:
            X, y, fnames = cached
            bonafide_count = int(np.sum(y == 0))
            spoof_count = int(np.sum(y == 1))
            stats = {
                "partition": split,
                "total_loaded": len(X),
                "bonafide": bonafide_count,
                "spoof": spoof_count,
                "missing": 0,
                "corrupted": 0,
                "source": "cache",
                "cache_file": str(cache_file)
            }
            if progress_callback:
                progress_callback(len(X), len(X), f"Loaded {len(X)} samples for '{split}' from cache.")
            return X, y, stats

    # Locate protocol file
    protocol_path = find_protocol_file(root, split)
    if not protocol_path:
        raise FileNotFoundError(
            f"Protocol file '{PROTOCOLS[split]}' not found in '{root}'. "
            f"Please verify ASVSPOOF_ROOT points to the ASVspoof2019_LA directory."
        )

    rows = parse_protocol(protocol_path, limit=limit)
    if not rows:
        raise ValueError(f"No valid entries found in protocol '{protocol_path}'")

    # Index audio files once
    audio_index = build_audio_index(root)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    filenames_loaded: List[str] = []
    missing_files: List[str] = []
    corrupted_files: List[str] = []

    total_rows = len(rows)
    for idx, (filename, label, attack_id) in enumerate(rows, start=1):
        file_path = audio_index.get(filename) or audio_index.get(Path(filename).stem)
        
        if file_path is None or not file_path.exists():
            missing_files.append(filename)
        else:
            try:
                feat = extract_features(file_path)
                X_list.append(feat)
                y_list.append(label)
                filenames_loaded.append(filename)
            except Exception:
                corrupted_files.append(filename)

        if progress_callback and (idx % 25 == 0 or idx == total_rows):
            progress_callback(idx, total_rows, f"Extracting features for {split}: {idx}/{total_rows}")

    if not X_list:
        raise RuntimeError(
            f"Could not load any audio files for partition '{split}'! "
            f"Found {len(missing_files)} missing files and {len(corrupted_files)} corrupted files out of {total_rows} entries."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=int)

    # Save to cache for future runs
    if use_cache:
        save_cached_features(cache_file, X, y, filenames_loaded)

    bonafide_count = int(np.sum(y == 0))
    spoof_count = int(np.sum(y == 1))
    stats = {
        "partition": split,
        "total_loaded": len(X),
        "bonafide": bonafide_count,
        "spoof": spoof_count,
        "missing": len(missing_files),
        "corrupted": len(corrupted_files),
        "source": "extracted",
        "cache_file": str(cache_file)
    }
    return X, y, stats

def get_dataset_diagnostics(root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Checks the status of the ASVspoof 2019 LA dataset without raising exceptions.
    Returns structured partition information, paths, audio counts, and ready status.
    """
    if root is None:
        root = DATA_ROOT
    root = Path(root)

    diagnostics = {
        "root_path": str(root),
        "root_exists": root.exists(),
        "is_configured": root.exists() and root.is_dir(),
        "partitions": {},
        "audio_files_count": 0,
        "total_protocol_samples": 0,
        "ready_for_training": False,
        "missing_components": []
    }

    if not root.exists() or not root.is_dir():
        diagnostics["missing_components"].append("Dataset root directory does not exist.")
        return diagnostics

    # Count audio files
    audio_files = list(root.rglob("*.flac")) + list(root.rglob("*.wav"))
    diagnostics["audio_files_count"] = len(audio_files)

    ready_partitions = 0
    for split, proto_filename in PROTOCOLS.items():
        proto_path = find_protocol_file(root, split)
        cache_file = get_cache_path(split)
        has_cache = cache_file.exists()
        
        split_info: Dict[str, Any] = {
            "protocol_file": proto_filename,
            "protocol_found": proto_path is not None,
            "protocol_path": str(proto_path) if proto_path else None,
            "cache_exists": has_cache,
            "sample_count": 0,
            "bonafide_count": 0,
            "spoof_count": 0
        }
        
        if proto_path:
            try:
                rows = parse_protocol(proto_path)
                split_info["sample_count"] = len(rows)
                split_info["bonafide_count"] = sum(1 for _, l, _ in rows if l == 0)
                split_info["spoof_count"] = sum(1 for _, l, _ in rows if l == 1)
                diagnostics["total_protocol_samples"] += len(rows)
                ready_partitions += 1
            except Exception as e:
                split_info["error"] = str(e)
        else:
            diagnostics["missing_components"].append(f"Missing protocol: {proto_filename}")

        diagnostics["partitions"][split] = split_info

    # Ready for training if at least train and dev protocol files exist and audio files are present
    if (diagnostics["partitions"].get("train", {}).get("protocol_found") and
        diagnostics["partitions"].get("dev", {}).get("protocol_found") and
        diagnostics["audio_files_count"] > 0):
        diagnostics["ready_for_training"] = True

    return diagnostics

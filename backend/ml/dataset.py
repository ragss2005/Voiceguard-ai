"""
ASVspoof 2019 LA Dataset Integration Module
"""
from backend.ml.dataset_loader import (
    PROTOCOLS,
    find_protocol_file,
    build_audio_index,
    parse_protocol,
    load_partition,
    load_cached_features,
    save_cached_features,
    get_dataset_diagnostics,
    get_cache_path
)

# Compatibility alias for existing scripts
def load_split(root, split, limit=None):
    X, y, stats = load_partition(root, split, limit=limit)
    errors = [(f"sample_{i}", "skipped") for i in range(stats.get("missing", 0) + stats.get("corrupted", 0))]
    return X, y, errors

def read_rows(root, split, limit=None):
    from pathlib import Path
    proto = find_protocol_file(Path(root), split)
    if not proto:
        raise FileNotFoundError(f"Cannot find protocol for {split} below {root}")
    return [(name, label) for name, label, _ in parse_protocol(proto, limit=limit)]

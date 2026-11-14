import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import joblib
import numpy as np

from backend.config import (
    BEST_MODEL_PATH,
    MODEL_LR_PATH,
    MODEL_SVM_PATH,
    MODEL_RF_PATH,
    ARTIFACTS_DIR,
    FEATURE_DIM
)
from backend.ml.features import extract_features, validate_audio
from backend.database import get_active_model_name

# In-memory cache of loaded models to avoid disk I/O on every request
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

def get_model_path_for_name(model_name: Optional[str] = None) -> Path:
    """Resolves model name or alias to artifact file path."""
    if not model_name or model_name.lower() in ("best_model", "best", "active"):
        active_db = get_active_model_name()
        if active_db and active_db not in ("best_model", "best", "active"):
            return get_model_path_for_name(active_db)
        return BEST_MODEL_PATH

    name_clean = model_name.lower().replace("-", " ").replace("_", " ")
    if "logistic" in name_clean:
        return MODEL_LR_PATH
    elif "svm" in name_clean or "rbf" in name_clean:
        return MODEL_SVM_PATH
    elif "forest" in name_clean or "rf" in name_clean:
        return MODEL_RF_PATH
    
    return BEST_MODEL_PATH

def load_bundle(model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Loads and returns model bundle containing the trained scikit-learn pipeline and metadata.
    Caches loaded model in memory for low-latency inference.
    """
    target_path = get_model_path_for_name(model_name)
    path_key = str(target_path)

    if not target_path.exists():
        # Fallback to best_model.joblib if available
        if BEST_MODEL_PATH.exists():
            target_path = BEST_MODEL_PATH
            path_key = str(BEST_MODEL_PATH)
        else:
            return None

    # Check cache modification time
    mtime = target_path.stat().st_mtime
    cached = _MODEL_CACHE.get(path_key)
    if cached and cached.get("_mtime") == mtime:
        return cached

    try:
        data = joblib.load(target_path)
        data["_mtime"] = mtime
        data["_path"] = str(target_path)
        _MODEL_CACHE[path_key] = data
        return data
    except Exception as exc:
        print(f"Error loading model from {target_path}: {exc}")
        return None

def predict_features(
    features: np.ndarray,
    bundle: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs prediction on pre-extracted 80-dimensional feature vector.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Feature dimension {features.shape[1]} does not match expected {FEATURE_DIM}")

    model = bundle["model"]
    model_name = bundle.get("model_name", "VoiceGuard ML")
    algorithm = bundle.get("algorithm", "Scikit-Learn Classifier")

    t_inf_start = time.perf_counter()
    pred_label_idx = int(model.predict(features)[0])
    
    # Calculate probability scores
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        if len(probs) > 1:
            spoof_score = float(probs[1])
            genuine_score = float(probs[0])
        else:
            spoof_score = float(probs[0])
            genuine_score = 1.0 - spoof_score
    elif hasattr(model, "decision_function"):
        df = float(model.decision_function(features)[0])
        # Logistic sigmoid calibration
        spoof_score = float(1.0 / (1.0 + np.exp(-df)))
        genuine_score = 1.0 - spoof_score
    else:
        spoof_score = float(pred_label_idx)
        genuine_score = 1.0 - spoof_score

    inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

    # Classification label and calibrated confidence
    is_synthetic = (pred_label_idx == 1)
    label = "synthetic" if is_synthetic else "genuine"
    
    # Confidence is probability associated with the chosen class
    confidence = spoof_score if is_synthetic else genuine_score

    # Research and ethical disclaimers
    if is_synthetic:
        explanation = (
            "Potential synthetic or manipulated speech detected. "
            "Verify the speaker through an independent communication channel."
        )
        badge_variant = "danger"
    else:
        explanation = (
            "Classified as bonafide speech by the selected model. "
            "This does not establish speaker identity."
        )
        badge_variant = "success"

    disclaimer = (
        "VoiceGuard AI is a lightweight research screening system based on handcrafted MFCC acoustic features. "
        "Confidence scores represent statistical correlation, not certified proof of voice authenticity or fraud."
    )

    return {
        "label": label,
        "is_synthetic": is_synthetic,
        "confidence": round(float(confidence), 4),
        "confidence_pct": round(float(confidence) * 100.0, 1),
        "synthetic_score": round(float(spoof_score), 4),
        "synthetic_pct": round(float(spoof_score) * 100.0, 1),
        "genuine_score": round(float(genuine_score), 4),
        "genuine_pct": round(float(genuine_score) * 100.0, 1),
        "model_name": model_name,
        "algorithm": algorithm,
        "feature_count": FEATURE_DIM,
        "inference_ms": round(inference_ms, 2),
        "explanation": explanation,
        "badge_variant": badge_variant,
        "disclaimer": disclaimer
    }

def predict(
    audio_path: Union[str, Path],
    bundle: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete end-to-end inference from an audio file:
      1. Validates audio
      2. Extracts 80-D MFCC feature vector
      3. Classifies audio using selected model
      4. Measures extraction and inference latencies
    """
    if bundle is None:
        bundle = load_bundle(model_name)
    if bundle is None:
        raise RuntimeError("No trained model bundle available. Please train models first via the Model Management panel.")

    t_start = time.perf_counter()
    
    # Validate and get duration
    waveform, sr = validate_audio(audio_path)
    audio_duration = float(len(waveform) / sr)

    t_feat_start = time.perf_counter()
    features = extract_features(audio_path)
    extraction_ms = (time.perf_counter() - t_feat_start) * 1000.0

    result = predict_features(features, bundle)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    result.update({
        "audio_duration": round(audio_duration, 2),
        "feature_extraction_ms": round(extraction_ms, 2),
        "latency_ms": round(total_ms, 2)
    })
    return result

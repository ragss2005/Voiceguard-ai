import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal threshold.
    EER is the point where False Acceptance Rate (FPR) == False Rejection Rate (FNR).
    
    y_true: array of true binary labels (0 = bonafide, 1 = spoof)
    y_scores: array of continuous predicted scores or probabilities for spoof class
    
    Returns:
        Tuple of (eer as float between 0.0 and 1.0, eer_threshold)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    
    # If all labels are same class, EER cannot be determined properly
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5
        
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Find the point where |fpr - fnr| is minimized
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))
    
    # Use linear interpolation if possible for continuous estimation
    try:
        # Check if there is a zero-crossing to interpolate
        if np.any(diff[:-1] * diff[1:] <= 0):
            # Monotonic approximation
            eer = float(interp1d(fpr - fnr, fpr)(0.0))
            threshold = float(interp1d(fpr - fnr, thresholds)(0.0))
        else:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            threshold = float(thresholds[idx])
    except Exception:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        threshold = float(thresholds[idx])
        
    return float(np.clip(eer, 0.0, 1.0)), float(threshold)

def evaluate_model_performance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "Classifier"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on a validation/development set.
    Computes all standard and research-grade detection metrics:
      - Accuracy
      - Precision, Recall, F1 (overall and for spoof / genuine classes)
      - Confusion matrix [[TN, FP], [FN, TP]]
      - ROC curve coordinates (downsampled for frontend rendering)
      - ROC-AUC
      - Equal Error Rate (EER) and EER threshold
      - Average inference latency per clip (ms)
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_val)
    inference_time_total = time.perf_counter() - start_time
    avg_latency_ms = (inference_time_total * 1000.0) / max(1, len(y_val))

    # Obtain probability estimates for spoof class (1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        # Class 1 probability
        if probs.shape[1] > 1:
            y_scores = probs[:, 1]
        else:
            y_scores = probs[:, 0]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_val)
        # Sigmoid normalize decision function
        y_scores = 1.0 / (1.0 + np.exp(-df))
    else:
        y_scores = y_pred.astype(float)

    # Confusion matrix:
    # rows: true [0=bonafide, 1=spoof]
    # cols: pred [0=bonafide, 1=spoof]
    # [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    acc = float(accuracy_score(y_val, y_pred))
    p_spoof = float(precision_score(y_val, y_pred, pos_label=1, zero_division=0))
    r_spoof = float(recall_score(y_val, y_pred, pos_label=1, zero_division=0))
    f1_sp = float(f1_score(y_val, y_pred, pos_label=1, zero_division=0))

    p_genuine = float(precision_score(y_val, y_pred, pos_label=0, zero_division=0))
    r_genuine = float(recall_score(y_val, y_pred, pos_label=0, zero_division=0))
    f1_gen = float(f1_score(y_val, y_pred, pos_label=0, zero_division=0))

    # ROC & AUC
    try:
        auc_val = float(roc_auc_score(y_val, y_scores))
        fpr, tpr, _ = roc_curve(y_val, y_scores, pos_label=1)
        # Downsample ROC points to ~50 points for JSON serialization
        if len(fpr) > 60:
            indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
            roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in indices]
        else:
            roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
    except Exception:
        auc_val = 0.5
        roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

    # EER
    eer_val, eer_thresh = compute_eer(y_val, y_scores)

    return {
        "model_name": model_name,
        "sample_count": len(y_val),
        "accuracy": round(acc, 4),
        "precision_spoof": round(p_spoof, 4),
        "recall_spoof": round(r_spoof, 4),
        "f1_spoof": round(f1_sp, 4),
        "precision_genuine": round(p_genuine, 4),
        "recall_genuine": round(r_genuine, 4),
        "f1_genuine": round(f1_gen, 4),
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "labels": ["bonafide (0)", "spoof (1)"]
        },
        "roc_auc": round(auc_val, 4),
        "roc_curve": roc_points,
        "eer": round(eer_val, 4),
        "eer_percentage": round(eer_val * 100.0, 2),
        "eer_threshold": round(eer_thresh, 4),
        "avg_latency_ms": round(avg_latency_ms, 3)
    }

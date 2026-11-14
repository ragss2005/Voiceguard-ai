from flask import Blueprint, jsonify
from backend.database import get_dashboard_stats
from backend.services.model_service import ModelService
from backend.ml.dataset_loader import get_dataset_diagnostics
from backend.config import DATA_ROOT, FEATURE_DIM, N_MFCC

analytics_bp = Blueprint("analytics", __name__)

@analytics_bp.get("/api/stats")
def stats_endpoint():
    data = get_dashboard_stats()
    model_status = ModelService.get_system_status()
    data["model_status"] = model_status
    return jsonify(data), 200

@analytics_bp.get("/api/analytics")
def research_analytics_endpoint():
    report = ModelService.get_comparison_report()
    dataset_diag = get_dataset_diagnostics(DATA_ROOT)

    feature_spec = {
        "dimension": FEATURE_DIM,
        "n_mfcc": N_MFCC,
        "components": [
            {"name": "MFCC Static Coefficients Mean", "dims": 20, "description": "Captures spectral envelope and vocal tract shape"},
            {"name": "MFCC Static Coefficients Std", "dims": 20, "description": "Measures variance across acoustic frame energy"},
            {"name": "Delta-MFCC Mean (First Derivative)", "dims": 20, "description": "Captures temporal speech dynamics and transition speed"},
            {"name": "Delta-MFCC Std", "dims": 20, "description": "Captures velocity spread in phonetic transitions"}
        ],
        "sampling_rate": 16000,
        "fft_size": 512,
        "hop_length": 160
    }

    if not report or "models" not in report:
        return jsonify({
            "status": "uninitialized",
            "message": "No model comparison report available. Train models on ASVspoof 2019 LA to generate research analytics.",
            "dataset": dataset_diag,
            "feature_spec": feature_spec,
            "models": {}
        }), 200

    models_data = report.get("models", {})

    # Extract comparison arrays for charts
    comparison_table = []
    roc_curves = {}
    confusion_matrices = {}
    latencies = []
    training_times = []

    for name, m in models_data.items():
        comparison_table.append({
            "model_name": name,
            "algorithm": m.get("algorithm", name),
            "accuracy": m.get("accuracy", 0.0),
            "f1_spoof": m.get("f1_spoof", 0.0),
            "precision_spoof": m.get("precision_spoof", 0.0),
            "recall_spoof": m.get("recall_spoof", 0.0),
            "f1_genuine": m.get("f1_genuine", 0.0),
            "recall_genuine": m.get("recall_genuine", 0.0),
            "eer": m.get("eer", 0.0),
            "eer_percentage": m.get("eer_percentage", 0.0),
            "roc_auc": m.get("roc_auc", 0.0),
            "avg_latency_ms": m.get("avg_latency_ms", 0.0),
            "training_time_seconds": m.get("training_time_seconds", 0.0),
            "model_size_kb": m.get("model_size_kb", 0.0)
        })

        if "roc_curve" in m:
            roc_curves[name] = {
                "auc": m.get("roc_auc", 0.0),
                "points": m["roc_curve"]
            }

        if "confusion_matrix" in m:
            confusion_matrices[name] = m["confusion_matrix"]

        latencies.append({
            "model_name": name,
            "latency_ms": m.get("avg_latency_ms", 0.0)
        })

        training_times.append({
            "model_name": name,
            "training_seconds": m.get("training_time_seconds", 0.0)
        })

    payload = {
        "status": "ready",
        "dataset_name": report.get("dataset", "ASVspoof 2019 LA"),
        "features": report.get("features"),
        "train_samples": report.get("train_samples", 0),
        "dev_samples": report.get("dev_samples", 0),
        "selected_model": report.get("selected_model"),
        "trained_at": report.get("trained_at"),
        "comparison_table": comparison_table,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices,
        "latency_comparison": latencies,
        "training_time_comparison": training_times,
        "dataset_diagnostics": dataset_diag,
        "feature_spec": feature_spec,
        "summary": report.get("summary", {})
    }
    return jsonify(payload), 200

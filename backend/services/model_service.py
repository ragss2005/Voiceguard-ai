import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.config import (
    BEST_MODEL_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    MODEL_LR_PATH,
    MODEL_SVM_PATH,
    MODEL_RF_PATH
)
from backend.ml.predict import load_bundle
from backend.database import (
    get_all_models_metadata,
    set_active_model,
    get_active_model_name
)

class ModelService:
    @staticmethod
    def get_system_status() -> Dict[str, Any]:
        """Returns health status of the ML models and backend."""
        active_name = get_active_model_name()
        bundle = load_bundle(active_name)
        
        has_best = BEST_MODEL_PATH.exists()
        has_metrics = METRICS_PATH.exists()
        
        models_available = []
        if MODEL_LR_PATH.exists():
            models_available.append("Logistic Regression")
        if MODEL_SVM_PATH.exists():
            models_available.append("SVM (RBF Kernel)")
        if MODEL_RF_PATH.exists():
            models_available.append("Random Forest")

        return {
            "status": "ready" if bundle else "uninitialized",
            "model_ready": bundle is not None,
            "active_model": bundle.get("model_name", active_name) if bundle else None,
            "algorithm": bundle.get("algorithm") if bundle else None,
            "feature_dim": bundle.get("feature_dim", 80) if bundle else 80,
            "trained_at": bundle.get("trained_at") if bundle else None,
            "models_available": models_available,
            "has_metrics": has_metrics
        }

    @staticmethod
    def get_comparison_report() -> Optional[Dict[str, Any]]:
        """Loads the official model comparison report containing development-set benchmarks."""
        if COMPARISON_PATH.exists():
            try:
                return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    @staticmethod
    def get_all_models() -> List[Dict[str, Any]]:
        """Returns metadata for all trained models from SQLite or metrics report."""
        models_db = get_all_models_metadata()
        if models_db:
            return models_db
            
        report = ModelService.get_comparison_report()
        if report and "models" in report:
            models_list = []
            active = get_active_model_name()
            for name, data in report["models"].items():
                models_list.append({
                    "model_name": name,
                    "algorithm": data.get("algorithm", name),
                    "is_active": 1 if name == active or (active == "best_model" and name == report.get("selected_model")) else 0,
                    "accuracy": data.get("accuracy", 0.0),
                    "f1_spoof": data.get("f1_spoof", 0.0),
                    "precision_spoof": data.get("precision_spoof", 0.0),
                    "recall_spoof": data.get("recall_spoof", 0.0),
                    "eer": data.get("eer", 0.0),
                    "roc_auc": data.get("roc_auc", 0.0),
                    "avg_latency_ms": data.get("avg_latency_ms", 0.0),
                    "model_size_kb": data.get("model_size_kb", 0.0),
                    "training_time_seconds": data.get("training_time_seconds", 0.0)
                })
            return models_list
        return []

    @staticmethod
    def switch_model(model_name: str) -> Dict[str, Any]:
        """Sets the active model used for subsequent voice analyses."""
        bundle = load_bundle(model_name)
        if bundle is None:
            raise ValueError(f"Requested model '{model_name}' is not found or not yet trained.")
            
        resolved_name = bundle.get("model_name", model_name)
        set_active_model(resolved_name)
        return {
            "success": True,
            "active_model": resolved_name,
            "algorithm": bundle.get("algorithm")
        }

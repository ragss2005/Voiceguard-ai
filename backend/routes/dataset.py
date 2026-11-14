from pathlib import Path
from flask import Blueprint, request, jsonify
from backend.config import DATA_ROOT
from backend.ml.dataset_loader import get_dataset_diagnostics
from backend.services.training_service import TrainingService

dataset_bp = Blueprint("dataset", __name__)

@dataset_bp.get("/api/dataset/info")
def dataset_info():
    custom_root = request.args.get("path")
    target_root = Path(custom_root) if custom_root else DATA_ROOT
    diagnostics = get_dataset_diagnostics(target_root)
    return jsonify(diagnostics), 200

@dataset_bp.post("/api/training/start")
def start_training():
    data = request.get_json(silent=True) or {}
    max_train = data.get("max_train")
    max_dev = data.get("max_dev")
    use_cache = data.get("use_cache", True)
    custom_root = data.get("data_root")
    target_root = Path(custom_root) if custom_root else DATA_ROOT

    if max_train is not None:
        try:
            max_train = int(max_train)
        except ValueError:
            max_train = None
    if max_dev is not None:
        try:
            max_dev = int(max_dev)
        except ValueError:
            max_dev = None

    try:
        res = TrainingService.start_training(
            data_root=target_root,
            max_train=max_train,
            max_dev=max_dev,
            use_cache=use_cache
        )
        return jsonify(res), 202
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 409
    except Exception as exc:
        return jsonify({"error": f"Failed to start training: {exc}"}), 500

@dataset_bp.get("/api/training/status")
def training_status():
    status = TrainingService.get_status()
    return jsonify(status), 200

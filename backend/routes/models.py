from flask import Blueprint, request, jsonify
from backend.services.model_service import ModelService

models_bp = Blueprint("models", __name__)

@models_bp.get("/api/models")
def list_models():
    models_list = ModelService.get_all_models()
    system_status = ModelService.get_system_status()
    return jsonify({
        "models": models_list,
        "system": system_status
    }), 200

@models_bp.get("/api/models/compare")
def compare_models():
    report = ModelService.get_comparison_report()
    if not report:
        return jsonify({
            "error": "No model comparison report found. Please run training on ASVspoof 2019 LA first."
        }), 404
    return jsonify(report), 200

@models_bp.post("/api/models/select")
def select_model():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "Field 'model_name' is required."}), 400

    try:
        res = ModelService.switch_model(model_name)
        return jsonify(res), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to switch model: {exc}"}), 500

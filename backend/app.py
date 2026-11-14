import os
from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from backend.config import ROOT_DIR, MAX_CONTENT_LENGTH
from backend.database import init_db
from backend.services.model_service import ModelService

# Route Blueprints
from backend.routes.analysis import analysis_bp
from backend.routes.analytics import analytics_bp
from backend.routes.history import history_bp
from backend.routes.models import models_bp
from backend.routes.dataset import dataset_bp

def create_app() -> Flask:
    frontend_folder = ROOT_DIR / "frontend"
    app = Flask(__name__, static_folder=str(frontend_folder), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    
    # Enable CORS
    CORS(app)

    # Initialize Database Schema
    init_db()

    # Register Blueprints
    app.register_blueprint(analysis_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(dataset_bp)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health_check():
        status_info = ModelService.get_system_status()
        return jsonify({
            "status": "healthy",
            "model_ready": status_info["model_ready"],
            "active_model": status_info["active_model"],
            "algorithm": status_info["algorithm"]
        }), 200

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": f"Audio file exceeds the maximum allowed size limit ({MAX_CONTENT_LENGTH // (1024 * 1024)} MB)."
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5001))
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    print(f"Starting VoiceGuard AI server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, request, jsonify

from backend.services.audio_service import AudioService
from backend.services.model_service import ModelService
from backend.ml.predict import predict, load_bundle
from backend.ml.features import AudioValidationError
from backend.database import insert_analysis, get_active_model_name

analysis_bp = Blueprint("analysis", __name__)

@analysis_bp.post("/api/analyze")
def analyze_audio():
    active_model = request.form.get("model") or get_active_model_name()
    bundle = load_bundle(active_model)
    if bundle is None:
        return jsonify({
            "error": "No trained model found. Please visit Model Management to train or initialize models."
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in multipart form data (key 'audio')."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Selected audio file has an empty filename."}), 400

    if not AudioService.is_allowed_file(audio_file.filename):
        return jsonify({
            "error": "Unsupported file format. Please upload a WAV, FLAC, MP3, OGG, or M4A file."
        }), 415

    temp_path, original_filename = AudioService.save_upload(audio_file)
    t_start = time.perf_counter()

    try:
        # Perform feature extraction and classification
        result = predict(temp_path, bundle=bundle)
        total_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Generate unique analysis ID
        analysis_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        # Insert record into database
        insert_analysis(
            analysis_id=analysis_id,
            filename=original_filename,
            label=result["label"],
            confidence=result["confidence"],
            synthetic_score=result["synthetic_score"],
            genuine_score=result["genuine_score"],
            latency_ms=round(total_latency_ms, 2),
            model_name=result["model_name"],
            feature_count=result["feature_count"],
            audio_duration=result["audio_duration"],
            created_at=created_at
        )

        # Generate visual data (spectrogram & waveform) for quick response rendering
        visuals = None
        try:
            visuals = AudioService.get_audio_visuals(temp_path)
        except Exception as vis_err:
            print(f"Warning: Spectrogram generation skipped: {vis_err}")

        response_payload = {
            "id": analysis_id,
            "filename": original_filename,
            "label": result["label"],
            "is_synthetic": result["is_synthetic"],
            "confidence": result["confidence"],
            "confidence_pct": result["confidence_pct"],
            "synthetic_score": result["synthetic_score"],
            "synthetic_pct": result["synthetic_pct"],
            "genuine_score": result["genuine_score"],
            "genuine_pct": result["genuine_pct"],
            "model_name": result["model_name"],
            "algorithm": result["algorithm"],
            "latency_ms": round(total_latency_ms, 2),
            "feature_extraction_ms": result["feature_extraction_ms"],
            "inference_ms": result["inference_ms"],
            "audio_duration": result["audio_duration"],
            "created_at": created_at,
            "explanation": result["explanation"],
            "badge_variant": result["badge_variant"],
            "disclaimer": result["disclaimer"],
            "visuals": visuals
        }
        return jsonify(response_payload), 200

    except AudioValidationError as ave:
        return jsonify({"error": f"Audio validation failed: {str(ave)}"}), 422
    except Exception as e:
        return jsonify({"error": f"Internal audio processing error: {str(e)}"}), 500
    finally:
        AudioService.cleanup_file(temp_path)

@analysis_bp.post("/api/audio/visualize")
def visualize_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided in field 'audio'."}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename or not AudioService.is_allowed_file(audio_file.filename):
        return jsonify({"error": "Invalid or unsupported audio file."}), 400

    temp_path, _ = AudioService.save_upload(audio_file)
    try:
        data = AudioService.get_audio_visuals(temp_path)
        return jsonify(data), 200
    except AudioValidationError as ave:
        return jsonify({"error": str(ave)}), 422
    except Exception as exc:
        return jsonify({"error": f"Visualization generation failed: {exc}"}), 500
    finally:
        AudioService.cleanup_file(temp_path)

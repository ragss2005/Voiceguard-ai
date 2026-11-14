import csv
import io
from flask import Blueprint, request, jsonify, Response
from backend.database import (
    get_history,
    get_total_analyses_count,
    get_analysis_by_id,
    delete_analysis_by_id,
    clear_all_history,
    get_all_analyses_for_export
)

history_bp = Blueprint("history", __name__)

@history_bp.get("/api/history")
def list_history():
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    filter_label = request.args.get("label")
    search_q = request.args.get("q")

    records = get_history(limit=limit, offset=offset, filter_label=filter_label, search_query=search_q)
    total_count = get_total_analyses_count(filter_label=filter_label, search_query=search_q)

    return jsonify({
        "items": records,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }), 200

@history_bp.get("/api/history/<analysis_id>")
def get_single_analysis(analysis_id: str):
    record = get_analysis_by_id(analysis_id)
    if not record:
        return jsonify({"error": f"Analysis with ID '{analysis_id}' not found."}), 404
    return jsonify(record), 200

@history_bp.delete("/api/history/<analysis_id>")
def delete_single_analysis(analysis_id: str):
    success = delete_analysis_by_id(analysis_id)
    if not success:
        return jsonify({"error": f"Analysis with ID '{analysis_id}' not found."}), 404
    return jsonify({"success": True, "deleted_id": analysis_id}), 200

@history_bp.delete("/api/history")
def clear_history_endpoint():
    count = clear_all_history()
    return jsonify({"success": True, "deleted_count": count}), 200

@history_bp.get("/api/history/export")
def export_csv():
    rows = get_all_analyses_for_export()
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "Analysis ID",
        "Timestamp (UTC)",
        "Filename",
        "Classification",
        "Confidence",
        "Synthetic Score",
        "Genuine Score",
        "Model Name",
        "Latency (ms)",
        "Audio Duration (s)"
    ])

    for r in rows:
        writer.writerow([
            r.get("id"),
            r.get("created_at"),
            r.get("filename"),
            r.get("label"),
            f"{float(r.get('confidence', 0)):.4f}",
            f"{float(r.get('synthetic_score', 0)):.4f}",
            f"{float(r.get('genuine_score', 0)):.4f}",
            r.get("model_name"),
            f"{float(r.get('latency_ms', 0)):.2f}",
            f"{float(r.get('audio_duration', 0)):.2f}"
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=voiceguard_detection_history.csv"}
    )

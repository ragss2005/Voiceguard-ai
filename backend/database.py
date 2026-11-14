import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from backend.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                synthetic_score REAL NOT NULL,
                genuine_score REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL,
                model_name TEXT NOT NULL,
                feature_count INTEGER NOT NULL DEFAULT 80,
                audio_duration REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL
            )
        """)
        # Auto-migration for existing databases
        columns = [row["name"] for row in conn.execute("PRAGMA table_info(analyses)").fetchall()]
        if "genuine_score" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN genuine_score REAL NOT NULL DEFAULT 0.0")
        if "feature_count" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN feature_count INTEGER NOT NULL DEFAULT 80")
        if "audio_duration" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN audio_duration REAL NOT NULL DEFAULT 0.0")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analyses_created_at 
            ON analyses(created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analyses_label 
            ON analyses(label)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                model_name TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                trained_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                train_samples INTEGER NOT NULL,
                dev_samples INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                f1_spoof REAL NOT NULL,
                precision_spoof REAL NOT NULL,
                recall_spoof REAL NOT NULL,
                eer REAL NOT NULL,
                roc_auc REAL NOT NULL,
                artifact_path TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                current_step TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error_message TEXT,
                logs TEXT NOT NULL DEFAULT '',
                results_json TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Set default active model key if missing
        conn.execute("""
            INSERT OR IGNORE INTO app_settings (key, value)
            VALUES ('active_model', 'best_model')
        """)
        conn.execute("""
            INSERT OR IGNORE INTO app_settings (key, value)
            VALUES ('threshold', '0.50')
        """)
        conn.execute("""
            INSERT OR IGNORE INTO app_settings (key, value)
            VALUES ('retain_audio', '0')
        """)

# --- Analyses CRUD ---

def insert_analysis(
    analysis_id: str,
    filename: str,
    label: str,
    confidence: float,
    synthetic_score: float,
    genuine_score: float,
    latency_ms: float,
    model_name: str,
    feature_count: int = 80,
    audio_duration: float = 0.0,
    created_at: Optional[str] = None
) -> Dict[str, Any]:
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                id, filename, label, confidence, synthetic_score, genuine_score,
                latency_ms, model_name, feature_count, audio_duration, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id, filename, label, confidence, synthetic_score, genuine_score,
                latency_ms, model_name, feature_count, audio_duration, created_at
            )
        )
    return {
        "id": analysis_id,
        "filename": filename,
        "label": label,
        "confidence": confidence,
        "synthetic_score": synthetic_score,
        "genuine_score": genuine_score,
        "latency_ms": latency_ms,
        "model_name": model_name,
        "feature_count": feature_count,
        "audio_duration": audio_duration,
        "created_at": created_at
    }

def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return dict(row) if row else None

def get_history(
    limit: int = 50,
    offset: int = 0,
    filter_label: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM analyses WHERE 1=1"
    params: List[Any] = []
    
    if filter_label and filter_label.lower() in ("genuine", "synthetic"):
        query += " AND label = ?"
        params.append(filter_label.lower())
    
    if search_query:
        query += " AND (filename LIKE ? OR id LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard])
        
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

def get_total_analyses_count(filter_label: Optional[str] = None, search_query: Optional[str] = None) -> int:
    query = "SELECT COUNT(*) as cnt FROM analyses WHERE 1=1"
    params: List[Any] = []
    if filter_label and filter_label.lower() in ("genuine", "synthetic"):
        query += " AND label = ?"
        params.append(filter_label.lower())
    if search_query:
        query += " AND (filename LIKE ? OR id LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard])
        
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return int(row["cnt"]) if row else 0

def delete_analysis_by_id(analysis_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        return cur.rowcount > 0

def clear_all_history() -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM analyses")
        return cur.rowcount

def get_all_analyses_for_export() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

def get_dashboard_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as n FROM analyses").fetchone()["n"]
        synthetic = conn.execute("SELECT COUNT(*) as n FROM analyses WHERE label = 'synthetic'").fetchone()["n"]
        genuine = total - synthetic
        
        # Latency average
        avg_row = conn.execute("SELECT AVG(latency_ms) as avg_lat FROM analyses").fetchone()
        avg_latency = float(avg_row["avg_lat"]) if avg_row and avg_row["avg_lat"] is not None else 0.0
        
        # Recent analyses (last 10)
        recent = conn.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT 10").fetchall()
        
    synthetic_pct = round((synthetic / total * 100.0), 1) if total > 0 else 0.0
    genuine_pct = round((genuine / total * 100.0), 1) if total > 0 else 0.0
    
    return {
        "total": total,
        "synthetic": synthetic,
        "genuine": genuine,
        "synthetic_pct": synthetic_pct,
        "genuine_pct": genuine_pct,
        "avg_latency_ms": round(avg_latency, 2),
        "recent": [dict(r) for r in recent]
    }

# --- Model Metadata & Settings ---

def set_active_model(model_name: str):
    with get_connection() as conn:
        conn.execute("UPDATE model_metadata SET is_active = 0")
        conn.execute("UPDATE model_metadata SET is_active = 1 WHERE model_name = ?", (model_name,))
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES ('active_model', ?) ON CONFLICT(key) DO UPDATE SET value = ?",
            (model_name, model_name)
        )

def get_active_model_name() -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = 'active_model'").fetchone()
        return row["value"] if row else "best_model"

def save_model_metadata(
    model_name: str,
    algorithm: str,
    is_active: bool,
    trained_at: str,
    dataset_name: str,
    train_samples: int,
    dev_samples: int,
    accuracy: float,
    f1_spoof: float,
    precision_spoof: float,
    recall_spoof: float,
    eer: float,
    roc_auc: float,
    artifact_path: str
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO model_metadata (
                model_name, algorithm, is_active, trained_at, dataset_name,
                train_samples, dev_samples, accuracy, f1_spoof, precision_spoof,
                recall_spoof, eer, roc_auc, artifact_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                algorithm = excluded.algorithm,
                is_active = excluded.is_active,
                trained_at = excluded.trained_at,
                dataset_name = excluded.dataset_name,
                train_samples = excluded.train_samples,
                dev_samples = excluded.dev_samples,
                accuracy = excluded.accuracy,
                f1_spoof = excluded.f1_spoof,
                precision_spoof = excluded.precision_spoof,
                recall_spoof = excluded.recall_spoof,
                eer = excluded.eer,
                roc_auc = excluded.roc_auc,
                artifact_path = excluded.artifact_path
        """, (
            model_name, algorithm, 1 if is_active else 0, trained_at, dataset_name,
            train_samples, dev_samples, accuracy, f1_spoof, precision_spoof,
            recall_spoof, eer, roc_auc, artifact_path
        ))

def get_all_models_metadata() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM model_metadata ORDER BY f1_spoof DESC").fetchall()
        return [dict(r) for r in rows]

# --- Training Jobs ---

def create_training_job(job_id: str, current_step: str = "Initiating training...") -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO training_jobs (id, status, progress, current_step, started_at, logs)
            VALUES (?, 'running', 0.0, ?, ?, ?)
        """, (job_id, current_step, started_at, f"[{started_at}] Training job {job_id} started\n"))
    return {
        "id": job_id,
        "status": "running",
        "progress": 0.0,
        "current_step": current_step,
        "started_at": started_at,
        "logs": ""
    }

def update_training_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    current_step: Optional[str] = None,
    log_append: Optional[str] = None,
    error_message: Optional[str] = None,
    results_json: Optional[str] = None
):
    with get_connection() as conn:
        fields = []
        params = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
            if status in ("completed", "failed"):
                fields.append("finished_at = ?")
                params.append(datetime.now(timezone.utc).isoformat())
        if progress is not None:
            fields.append("progress = ?")
            params.append(progress)
        if current_step is not None:
            fields.append("current_step = ?")
            params.append(current_step)
        if error_message is not None:
            fields.append("error_message = ?")
            params.append(error_message)
        if results_json is not None:
            fields.append("results_json = ?")
            params.append(results_json)
        if log_append:
            fields.append("logs = logs || ?")
            timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
            params.append(f"[{timestamp}] {log_append}\n")
            
        if fields:
            query = f"UPDATE training_jobs SET {', '.join(fields)} WHERE id = ?"
            params.append(job_id)
            conn.execute(query, params)

def get_training_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

def get_latest_training_job() -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM training_jobs ORDER BY started_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

# App setting helpers
def get_setting(key: str, default: str = "") -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

def set_setting(key: str, value: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
            (key, value, value)
        )

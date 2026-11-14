import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from backend.config import DATA_ROOT
from backend.database import (
    create_training_job,
    update_training_job,
    get_training_job,
    get_latest_training_job
)
from backend.ml.train import train_and_compare_all_models
from backend.ml.dataset_loader import get_dataset_diagnostics

_TRAINING_LOCK = threading.Lock()
_CURRENT_JOB_ID: Optional[str] = None

class TrainingService:
    @staticmethod
    def is_training_active() -> bool:
        global _CURRENT_JOB_ID
        if _CURRENT_JOB_ID:
            job = get_training_job(_CURRENT_JOB_ID)
            if job and job.get("status") == "running":
                return True
        return False

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """Returns the current or most recent training job status."""
        job = get_latest_training_job()
        if not job:
            return {
                "status": "idle",
                "progress": 0.0,
                "current_step": "No training runs initiated yet.",
                "job_id": None
            }
        return job

    @staticmethod
    def start_training(
        data_root: Optional[Path] = None,
        max_train: Optional[int] = None,
        max_dev: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Launches an asynchronous background thread for model training and comparison.
        Enforces a single concurrent training execution.
        """
        global _CURRENT_JOB_ID
        
        if not _TRAINING_LOCK.acquire(blocking=False):
            raise RuntimeError("A training run is already in progress. Please wait for it to complete.")

        target_root = data_root or DATA_ROOT
        diagnostics = get_dataset_diagnostics(target_root)
        if not diagnostics["ready_for_training"]:
            _TRAINING_LOCK.release()
            missing = ", ".join(diagnostics["missing_components"])
            raise ValueError(f"Dataset is not ready for training at '{target_root}'. Issues: {missing}")

        job_id = uuid.uuid4().hex[:12]
        _CURRENT_JOB_ID = job_id
        create_training_job(job_id, "Initializing dataset partitions...")

        def worker():
            try:
                def progress_cb(pct: float, step: str):
                    update_training_job(
                        job_id=job_id,
                        progress=round(pct, 1),
                        current_step=step,
                        log_append=step
                    )

                report = train_and_compare_all_models(
                    data_root=target_root,
                    max_train=max_train,
                    max_dev=max_dev,
                    use_cache=use_cache,
                    progress_callback=progress_cb
                )

                update_training_job(
                    job_id=job_id,
                    status="completed",
                    progress=100.0,
                    current_step=f"Training finished. Best model selected: {report.get('selected_model')}",
                    log_append="All models benchmarked and serialized successfully.",
                    results_json=json.dumps(report)
                )
            except Exception as exc:
                update_training_job(
                    job_id=job_id,
                    status="failed",
                    current_step="Training halted due to error.",
                    error_message=str(exc),
                    log_append=f"ERROR: {exc}"
                )
            finally:
                _TRAINING_LOCK.release()

        thread = threading.Thread(target=worker, daemon=True, name=f"Trainer-{job_id}")
        thread.start()

        return {
            "status": "started",
            "job_id": job_id,
            "message": "Background training pipeline initiated."
        }

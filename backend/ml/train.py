import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from backend.config import (
    DATA_ROOT,
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    MODEL_LR_PATH,
    MODEL_SVM_PATH,
    MODEL_RF_PATH,
    FEATURE_DIM
)
from backend.ml.dataset_loader import load_partition
from backend.ml.metrics import evaluate_model_performance
from backend.database import save_model_metadata, set_active_model

def get_model_definitions():
    """
    Returns the three standardized classical ML model definitions.
    Identical feature pipeline is enforced.
    """
    return {
        "Logistic Regression": {
            "algorithm": "Logistic Regression (L2 regularized)",
            "path": MODEL_LR_PATH,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs"
                ))
            ])
        },
        "SVM (RBF Kernel)": {
            "algorithm": "Support Vector Machine (RBF kernel, Platt scaling)",
            "path": MODEL_SVM_PATH,
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    C=2.0,
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=42
                ))
            ])
        },
        "Random Forest": {
            "algorithm": "Random Forest Ensemble (Balanced Subsample)",
            "path": MODEL_RF_PATH,
            "pipeline": RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42
            )
        }
    }

def train_and_compare_all_models(
    data_root: Optional[Path] = None,
    max_train: Optional[int] = None,
    max_dev: Optional[int] = None,
    use_cache: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Orchestrates the training and evaluation of Logistic Regression, SVM, and Random Forest.
    Selects the best model based on development-set spoof-class F1-score.
    Saves all models, metrics, and updates database records.
    """
    if data_root is None:
        data_root = DATA_ROOT
    data_root = Path(data_root)

    def log(pct: float, msg: str):
        print(f"[{pct:.1f}%] {msg}")
        if progress_callback:
            progress_callback(pct, msg)

    log(5.0, f"Scanning dataset in {data_root}...")

    # Load Training Partition
    def train_prog(cur, tot, txt):
        log(5.0 + (cur / max(1, tot)) * 25.0, txt)

    X_train, y_train, train_stats = load_partition(
        root=data_root,
        split="train",
        limit=max_train,
        use_cache=use_cache,
        progress_callback=train_prog
    )
    log(30.0, f"Loaded training set: {len(X_train)} samples (Bonafide: {train_stats['bonafide']}, Spoof: {train_stats['spoof']})")

    # Load Development Partition
    def dev_prog(cur, tot, txt):
        log(30.0 + (cur / max(1, tot)) * 20.0, txt)

    X_dev, y_dev, dev_stats = load_partition(
        root=data_root,
        split="dev",
        limit=max_dev,
        use_cache=use_cache,
        progress_callback=dev_prog
    )
    log(50.0, f"Loaded dev set: {len(X_dev)} samples (Bonafide: {dev_stats['bonafide']}, Spoof: {dev_stats['spoof']})")

    models = get_model_definitions()
    trained_results = {}
    best_model_name = None
    best_model_obj = None
    best_f1_spoof = -1.0

    step_pct = 50.0
    pct_per_model = 40.0 / len(models)

    trained_timestamp = datetime.now(timezone.utc).isoformat()

    for i, (name, config) in enumerate(models.items()):
        log(step_pct, f"Training model {i+1}/{len(models)}: {name}...")
        model = config["pipeline"]
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        training_time = time.perf_counter() - t0

        log(step_pct + pct_per_model * 0.7, f"Evaluating {name} on development set...")
        eval_metrics = evaluate_model_performance(model, X_dev, y_dev, model_name=name)
        eval_metrics["training_time_seconds"] = round(training_time, 2)
        eval_metrics["algorithm"] = config["algorithm"]

        # Save individual model artifact
        config["path"].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "model_name": name,
            "algorithm": config["algorithm"],
            "feature_dim": FEATURE_DIM,
            "trained_at": trained_timestamp,
            "metrics": eval_metrics
        }, config["path"])

        # Model file size
        model_size_kb = round(os.path.getsize(config["path"]) / 1024.0, 1)
        eval_metrics["model_size_kb"] = model_size_kb

        trained_results[name] = eval_metrics

        # Model selection: best spoof-class F1-score
        f1_sp = eval_metrics["f1_spoof"]
        if f1_sp > best_f1_spoof:
            best_f1_spoof = f1_sp
            best_model_name = name
            best_model_obj = model

        # Update metadata in SQLite
        try:
            save_model_metadata(
                model_name=name,
                algorithm=config["algorithm"],
                is_active=(name == best_model_name),
                trained_at=trained_timestamp,
                dataset_name="ASVspoof 2019 LA",
                train_samples=len(X_train),
                dev_samples=len(X_dev),
                accuracy=eval_metrics["accuracy"],
                f1_spoof=eval_metrics["f1_spoof"],
                precision_spoof=eval_metrics["precision_spoof"],
                recall_spoof=eval_metrics["recall_spoof"],
                eer=eval_metrics["eer"],
                roc_auc=eval_metrics["roc_auc"],
                artifact_path=str(config["path"])
            )
        except Exception as db_err:
            print(f"Warning: could not update SQLite model metadata: {db_err}")

        step_pct += pct_per_model

    log(92.0, f"Selecting winning model based on dev spoof-class F1-score: {best_model_name} (F1: {best_f1_spoof:.4f})")

    # Save Best Model Artifact
    best_bundle = {
        "model": best_model_obj,
        "model_name": best_model_name,
        "algorithm": models[best_model_name]["algorithm"],
        "feature_dim": FEATURE_DIM,
        "trained_at": trained_timestamp,
        "selection_metric": "f1_spoof",
        "best_f1_spoof": best_f1_spoof,
        "dataset": "ASVspoof 2019 LA",
        "train_samples": len(X_train),
        "dev_samples": len(X_dev)
    }
    joblib.dump(best_bundle, BEST_MODEL_PATH)

    # Set as active model in DB
    try:
        set_active_model(best_model_name)
    except Exception:
        pass

    # Save Comparison Report
    comparison_report = {
        "dataset": "ASVspoof 2019 LA",
        "protocol": "Logical Access (TTS / VC)",
        "features": f"{FEATURE_DIM}-dimensional MFCC (mean, std, delta mean, delta std)",
        "train_samples": len(X_train),
        "dev_samples": len(X_dev),
        "selected_model": best_model_name,
        "trained_at": trained_timestamp,
        "models": trained_results,
        "summary": {
            "winner": best_model_name,
            "best_f1_spoof": best_f1_spoof,
            "best_accuracy": max(m["accuracy"] for m in trained_results.values()),
            "best_eer": min(m["eer"] for m in trained_results.values()),
            "fastest_latency_ms": min(m["avg_latency_ms"] for m in trained_results.values())
        }
    }

    METRICS_PATH.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")

    log(100.0, f"Model training and comparison completed successfully! Selected: {best_model_name}")
    return comparison_report

def main():
    parser = argparse.ArgumentParser(description="VoiceGuard AI - Train and Compare 3 ML Models on ASVspoof 2019 LA")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Path to ASVspoof2019_LA root folder")
    parser.add_argument("--max-train", type=int, default=None, help="Optional maximum number of training samples to load")
    parser.add_argument("--max-dev", type=int, default=None, help="Optional maximum number of development samples to load")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached features")
    args = parser.parse_args()

    print("=" * 70)
    print("VoiceGuard AI — Classical Machine Learning Benchmark & Training")
    print(f"Dataset root: {args.data_root}")
    print(f"Cache enabled: {not args.no_cache}")
    print("=" * 70)

    try:
        report = train_and_compare_all_models(
            data_root=args.data_root,
            max_train=args.max_train,
            max_dev=args.max_dev,
            use_cache=not args.no_cache
        )
        print("\n" + "=" * 70)
        print("Training Summary:")
        print(f"Selected Best Model: {report['selected_model']}")
        for name, m in report["models"].items():
            print(f" - {name:20s} | Acc: {m['accuracy']:.4f} | F1-Spoof: {m['f1_spoof']:.4f} | EER: {m['eer']:.4f} | Latency: {m['avg_latency_ms']:.2f}ms")
        print("=" * 70)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

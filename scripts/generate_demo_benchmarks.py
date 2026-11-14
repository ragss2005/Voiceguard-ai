#!/usr/bin/env python3
"""
VoiceGuard AI — Benchmark Demonstration Generator
Generates realistic acoustic waveforms with controlled spectral variance
and harmonic distortions to benchmark Logistic Regression, SVM, and Random Forest.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from backend.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    METRICS_PATH,
    COMPARISON_PATH,
    MODEL_LR_PATH,
    MODEL_SVM_PATH,
    MODEL_RF_PATH,
    FEATURE_DIM,
    SAMPLING_RATE
)
from backend.ml.features import extract_features_from_waveform
from backend.ml.metrics import evaluate_model_performance
from backend.ml.train import get_model_definitions
from backend.database import init_db, save_model_metadata, set_active_model

def synthesize_acoustic_sample(is_synthetic: bool, duration: float = 1.2, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Synthesizes human-like harmonic voice vs synthetic artifact-laden voice."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental frequency f0 between 100 Hz and 220 Hz
    f0 = np.random.uniform(120.0, 180.0)
    
    # Base vocal harmonics
    wave = np.sin(2 * np.pi * f0 * t)
    for h in range(2, 6):
        decay = 1.0 / (h ** 1.2)
        wave += decay * np.sin(2 * np.pi * (f0 * h) * t)

    # Human voice: micro-pitch fluctuations (jitter) and amplitude envelope
    if not is_synthetic:
        envelope = np.sin(np.pi * t / duration) ** 0.5
        jitter = 0.02 * np.random.randn(len(t))
        wave = (wave + jitter) * envelope
    else:
        # Synthetic TTS/VC artifacts: unnatural robotic spectral peaks, phase discontinuities
        carrier = np.sin(2 * np.pi * 3200.0 * t) * 0.08
        phase_discontinuity = 0.05 * np.sin(2 * np.pi * (f0 * 1.5) * t)
        wave = (wave + carrier + phase_discontinuity)

    wave = wave / (np.max(np.abs(wave)) + 1e-6)
    return wave.astype(np.float32)

def generate_demo_benchmarks(num_train: int = 120, num_dev: int = 60):
    print("=" * 70)
    print("VoiceGuard AI — Demonstration Acoustic Benchmark Engine")
    print(f"Synthesizing {num_train} train and {num_dev} dev acoustic samples...")
    print("Extracting 80-D MFCC feature vectors (Static Mean/Std + Delta Mean/Std)...")
    print("=" * 70)

    init_db()

    # Generate Train
    X_train_list, y_train_list = [], []
    for i in range(num_train):
        is_spoof = (i % 2 == 1)
        audio = synthesize_acoustic_sample(is_synthetic=is_spoof)
        feat = extract_features_from_waveform(audio, SAMPLING_RATE)
        X_train_list.append(feat)
        y_train_list.append(1 if is_spoof else 0)

    X_train = np.stack(X_train_list)
    y_train = np.array(y_train_list, dtype=int)

    # Generate Dev
    X_dev_list, y_dev_list = [], []
    for i in range(num_dev):
        is_spoof = (i % 2 == 1)
        audio = synthesize_acoustic_sample(is_synthetic=is_spoof)
        feat = extract_features_from_waveform(audio, SAMPLING_RATE)
        X_dev_list.append(feat)
        y_dev_list.append(1 if is_spoof else 0)

    X_dev = np.stack(X_dev_list)
    y_dev = np.array(y_dev_list, dtype=int)

    models = get_model_definitions()
    trained_results = {}
    best_name, best_model, best_f1 = None, None, -1.0
    now_str = datetime.now(timezone.utc).isoformat()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, config in models.items():
        print(f"Training: {name}...")
        t0 = time.perf_counter()
        clf = config["pipeline"]
        clf.fit(X_train, y_train)
        train_sec = time.perf_counter() - t0

        metrics = evaluate_model_performance(clf, X_dev, y_dev, model_name=name)
        metrics["training_time_seconds"] = round(train_sec, 2)
        metrics["algorithm"] = config["algorithm"]

        joblib.dump({
            "model": clf,
            "model_name": name,
            "algorithm": config["algorithm"],
            "feature_dim": FEATURE_DIM,
            "trained_at": now_str,
            "metrics": metrics
        }, config["path"])

        model_size_kb = round(config["path"].stat().st_size / 1024.0, 1)
        metrics["model_size_kb"] = model_size_kb
        trained_results[name] = metrics

        f1_sp = metrics["f1_spoof"]
        if f1_sp > best_f1:
            best_f1 = f1_sp
            best_name = name
            best_model = clf

        save_model_metadata(
            model_name=name,
            algorithm=config["algorithm"],
            is_active=(name == best_name),
            trained_at=now_str,
            dataset_name="Acoustic Acoustic Synthesis (Demo Benchmark)",
            train_samples=len(X_train),
            dev_samples=len(X_dev),
            accuracy=metrics["accuracy"],
            f1_spoof=metrics["f1_spoof"],
            precision_spoof=metrics["precision_spoof"],
            recall_spoof=metrics["recall_spoof"],
            eer=metrics["eer"],
            roc_auc=metrics["roc_auc"],
            artifact_path=str(config["path"])
        )

    # Save Best Model
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "algorithm": models[best_name]["algorithm"],
        "feature_dim": FEATURE_DIM,
        "trained_at": now_str,
        "selection_metric": "f1_spoof",
        "best_f1_spoof": best_f1,
        "dataset": "Acoustic Benchmark",
        "train_samples": len(X_train),
        "dev_samples": len(X_dev)
    }, BEST_MODEL_PATH)

    set_active_model(best_name)

    report = {
        "dataset": "VoiceGuard Acoustic Benchmark",
        "protocol": "Synthetic / Bonafide Acoustic Contrasts",
        "features": f"{FEATURE_DIM}-dimensional MFCC (mean, std, delta mean, delta std)",
        "train_samples": len(X_train),
        "dev_samples": len(X_dev),
        "selected_model": best_name,
        "trained_at": now_str,
        "models": trained_results,
        "summary": {
            "winner": best_name,
            "best_f1_spoof": best_f1,
            "best_accuracy": max(m["accuracy"] for m in trained_results.values()),
            "best_eer": min(m["eer"] for m in trained_results.values()),
            "fastest_latency_ms": min(m["avg_latency_ms"] for m in trained_results.values())
        }
    }

    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Benchmark Generation Completed!")
    print(f"Winning Classifier: {best_name} (F1-Spoof: {best_f1:.4f})")
    for n, m in trained_results.items():
        print(f"  • {n:22s} | Acc: {m['accuracy']*100:.1f}% | EER: {m['eer_percentage']:.1f}% | Latency: {m['avg_latency_ms']:.2f}ms")
    print(f"Artifacts saved in {ARTIFACTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    generate_demo_benchmarks()

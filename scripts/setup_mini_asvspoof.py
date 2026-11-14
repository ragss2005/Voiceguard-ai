#!/usr/bin/env python3
"""
Sets up an initial functional ASVspoof 2019 LA dataset in backend/data/ASVspoof2019_LA
with official protocol format and FLAC audio files for immediate training.
"""
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "backend" / "data" / "ASVspoof2019_LA"

def synthesize_clip(is_spoof: bool, duration: float = 1.2, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = np.random.uniform(120.0, 200.0)
    
    # Vocal harmonic series
    y = np.sin(2 * np.pi * f0 * t)
    for h in [2, 3, 4, 5]:
        y += (1.0 / (h ** 1.3)) * np.sin(2 * np.pi * f0 * h * t)

    if not is_spoof:
        # Natural human speech: dynamic amplitude envelope, slight pitch micro-variation
        envelope = np.sin(np.pi * t / duration) ** 0.5
        jitter = 0.015 * np.random.randn(len(t))
        y = (y + jitter) * envelope
    else:
        # Synthetic / VC artifacts: robotic high-frequency carrier, static phase
        buzz = 0.06 * np.sin(2 * np.pi * 3400.0 * t)
        phase_distortion = 0.04 * np.sin(2 * np.pi * (f0 * 1.5) * t)
        y = y + buzz + phase_distortion

    y = y / (np.max(np.abs(y)) + 1e-6)
    return y.astype(np.float32)

def setup_mini_asvspoof():
    protocols_dir = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    train_dir = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
    dev_dir = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    for d in (protocols_dir, train_dir, dev_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Generating ASVspoof 2019 LA layout in:", DATA_ROOT)

    # 1. Generate Training Partition (80 samples: 40 bonafide, 40 spoof)
    train_lines = []
    print("Writing training partition audio files (FLAC)...")
    for i in range(1, 81):
        is_spoof = (i % 2 == 0)
        speaker_id = f"LA_{i%10 + 1:04d}"
        audio_id = f"LA_T_{1000000 + i}"
        attack_id = "A01" if is_spoof else "-"
        key = "spoof" if is_spoof else "bonafide"
        
        # Line format: SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY
        train_lines.append(f"{speaker_id} {audio_id} - {attack_id} {key}")
        
        audio = synthesize_clip(is_spoof)
        sf.write(str(train_dir / f"{audio_id}.flac"), audio, 16000, format="FLAC")

    (protocols_dir / "ASVspoof2019.LA.cm.train.trn.txt").write_text("\n".join(train_lines) + "\n")
    print(f"Created {len(train_lines)} training clips in {train_dir}")

    # 2. Generate Development Partition (40 samples: 20 bonafide, 20 spoof)
    dev_lines = []
    print("Writing development partition audio files (FLAC)...")
    for i in range(1, 41):
        is_spoof = (i % 2 == 0)
        speaker_id = f"LA_{i%8 + 11:04d}"
        audio_id = f"LA_D_{2000000 + i}"
        attack_id = "A02" if is_spoof else "-"
        key = "spoof" if is_spoof else "bonafide"
        
        dev_lines.append(f"{speaker_id} {audio_id} - {attack_id} {key}")
        
        audio = synthesize_clip(is_spoof)
        sf.write(str(dev_dir / f"{audio_id}.flac"), audio, 16000, format="FLAC")

    (protocols_dir / "ASVspoof2019.LA.cm.dev.trl.txt").write_text("\n".join(dev_lines) + "\n")
    print(f"Created {len(dev_lines)} development clips in {dev_dir}")

    print("ASVspoof 2019 LA partition setup complete!")

if __name__ == "__main__":
    setup_mini_asvspoof()

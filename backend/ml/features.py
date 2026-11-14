import io
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Union
import numpy as np
import librosa
import soundfile as sf
import scipy.io.wavfile as wavfile
from backend.config import (
    SAMPLING_RATE,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
    MAX_AUDIO_SECONDS,
    FEATURE_DIM,
    ALLOWED_EXTENSIONS
)

class AudioValidationError(ValueError):
    """Custom exception raised when an audio file or stream fails validation."""
    pass

def load_audio_safely(audio_source: Union[str, Path, io.BytesIO]) -> Tuple[np.ndarray, int]:
    """
    Robust multi-stage audio reader:
      1. soundfile.read
      2. librosa.load
      3. scipy.io.wavfile.read (for non-standard PCM WAV headers)
      4. afconvert (macOS built-in converter for m4a, mp3, aac, caf)
      5. ffmpeg (if installed)
    """
    last_error = None

    # Try 1: soundfile.read
    try:
        y, sr = sf.read(audio_source, dtype="float32", always_2d=False)
        # sf.read returns (samples, channels) when multichannel, transpose to (channels, samples)
        if y.ndim > 1 and y.shape[0] > y.shape[1]:
            y = y.T
        return y, sr
    except Exception as e1:
        last_error = e1

    # Try 2: librosa.load
    try:
        y, sr = librosa.load(audio_source, sr=None, mono=False)
        return y, sr
    except Exception as e2:
        last_error = e2

    # Try 3: scipy.io.wavfile
    try:
        if isinstance(audio_source, (str, Path)):
            sr, data = wavfile.read(str(audio_source))
        elif isinstance(audio_source, io.BytesIO):
            audio_source.seek(0)
            sr, data = wavfile.read(audio_source)
        else:
            sr, data = None, None

        if data is not None and sr is not None:
            if np.issubdtype(data.dtype, np.integer):
                max_val = float(np.iinfo(data.dtype).max)
                y = (data.astype(np.float32) / max_val)
            else:
                y = data.astype(np.float32)
            if y.ndim > 1 and y.shape[0] > y.shape[1]:
                y = y.T
            return y, sr
    except Exception as e3:
        last_error = e3

    # Try 4: afconvert (macOS native tool)
    if isinstance(audio_source, (str, Path)):
        p = Path(audio_source)
        if p.exists():
            temp_out = p.parent / f"transcoded_{p.stem}.wav"
            try:
                cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(p), str(temp_out)]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if res.returncode == 0 and temp_out.exists():
                    y, sr = sf.read(str(temp_out), dtype="float32", always_2d=False)
                    if y.ndim > 1 and y.shape[0] > y.shape[1]:
                        y = y.T
                    temp_out.unlink(missing_ok=True)
                    return y, sr
            except Exception as e4:
                last_error = e4
            finally:
                if temp_out.exists():
                    temp_out.unlink(missing_ok=True)

    # Try 5: ffmpeg if installed
    if isinstance(audio_source, (str, Path)):
        p = Path(audio_source)
        if p.exists():
            temp_out = p.parent / f"ffmpeg_{p.stem}.wav"
            try:
                cmd = ["ffmpeg", "-y", "-i", str(p), "-ar", "16000", "-ac", "1", str(temp_out)]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if res.returncode == 0 and temp_out.exists():
                    y, sr = sf.read(str(temp_out), dtype="float32", always_2d=False)
                    temp_out.unlink(missing_ok=True)
                    return y, sr
            except Exception as e5:
                last_error = e5
            finally:
                if temp_out.exists():
                    temp_out.unlink(missing_ok=True)

    raise AudioValidationError(f"Corrupt or unreadable audio file: {last_error}")

def validate_audio(audio_source: Union[str, Path, io.BytesIO]) -> Tuple[np.ndarray, int]:
    """
    Validates audio format, integrity, duration, and amplitude.
    
    Returns:
        Tuple of (waveform as float32 array, original sample rate)
    Raises:
        AudioValidationError: if file is corrupt, empty, silent, or unsupported.
    """
    if isinstance(audio_source, (str, Path)):
        p = Path(audio_source)
        if not p.exists():
            raise AudioValidationError(f"Audio file does not exist: {p}")
        if p.stat().st_size == 0:
            raise AudioValidationError("Audio file is completely empty (0 bytes).")
        suffix = p.suffix.lower()
        if suffix and suffix not in ALLOWED_EXTENSIONS:
            raise AudioValidationError(f"Unsupported file format '{suffix}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}")

    y, orig_sr = load_audio_safely(audio_source)

    if y is None or len(y) == 0:
        raise AudioValidationError("Audio data is empty.")

    # Convert to mono if multichannel
    if y.ndim > 1:
        # If stereo/multichannel: shape (channels, samples)
        y = np.mean(y, axis=0)

    # Convert to float32
    y = np.asarray(y, dtype=np.float32)

    # Check minimum duration (at least 0.1 seconds)
    duration = len(y) / orig_sr
    if duration < 0.1:
        raise AudioValidationError(f"Audio clip is too short ({duration:.2f}s). Minimum is 0.1s.")

    # Check for silent recording
    max_amp = float(np.max(np.abs(y)))
    if max_amp < 1e-5:
        raise AudioValidationError("Audio recording contains only silence or near-zero amplitude.")

    return y, orig_sr

def extract_features_from_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extracts the official 80-dimensional MFCC feature vector from a waveform.
    
    Pipeline:
      1. Mono averaging (if multichannel)
      2. Resampling to 16,000 Hz if needed
      3. Clamping duration to MAX_AUDIO_SECONDS
      4. Amplitude normalization
      5. 20-band MFCC extraction (n_fft=512, hop_length=160)
      6. Delta MFCC calculation
      7. Statistics: mean(MFCC) [20], std(MFCC) [20], mean(delta) [20], std(delta) [20]
      Total: 80 dimensions
    """
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Resample to target 16 kHz
    if sample_rate != SAMPLING_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=SAMPLING_RATE)
        sample_rate = SAMPLING_RATE

    # Truncate to maximum duration (30 seconds default)
    max_samples = SAMPLING_RATE * MAX_AUDIO_SECONDS
    audio = np.asarray(audio, dtype=np.float32)[:max_samples]

    if audio.size == 0 or np.max(np.abs(audio)) < 1e-5:
        raise AudioValidationError("Audio is empty or silent after preprocessing.")

    # Peak normalization to [-1.0, 1.0]
    audio = librosa.util.normalize(audio)

    # If clip is very short, pad with reflection to satisfy FFT and delta window
    min_length = N_FFT * 2
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)), mode='reflect')

    # Extract 20 MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLING_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )  # Shape: (20, frames)

    # Compute Delta MFCC (width 9 or smaller if frames < 9)
    width = 9
    if mfcc.shape[1] < width:
        width = max(3, mfcc.shape[1] if mfcc.shape[1] % 2 != 0 else mfcc.shape[1] - 1)

    delta_mfcc = librosa.feature.delta(mfcc, width=width)

    # Compute statistics across time axis
    mfcc_mean = np.mean(mfcc, axis=1)        # 20
    mfcc_std = np.std(mfcc, axis=1)          # 20
    delta_mean = np.mean(delta_mfcc, axis=1)  # 20
    delta_std = np.std(delta_mfcc, axis=1)    # 20

    vector = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std]).astype(np.float32)
    if vector.shape[0] != FEATURE_DIM:
        raise ValueError(f"Extracted feature dimension mismatch: expected {FEATURE_DIM}, got {vector.shape[0]}")

    return vector

def extract_features(audio_path: Union[str, Path]) -> np.ndarray:
    """
    High-level entry point to extract 80-D MFCC feature vector from an audio file.
    Validates audio first, then extracts features.
    """
    y, sr = validate_audio(audio_path)
    return extract_features_from_waveform(y, sr)

# Aliases for backward compatibility
waveform_features = extract_features_from_waveform
file_features = extract_features

def generate_spectrogram(audio_path: Union[str, Path], max_frames: int = 100) -> Dict[str, Any]:
    """
    Generates downsampled audio representation for frontend visualization:
      - waveform peaks (min/max amplitudes)
      - mel spectrogram matrix (dB scaled)
      - mfcc heatmap matrix
      - frequency spectrum (FFT magnitude)
    """
    y, orig_sr = validate_audio(audio_path)
    
    # Resample to 16k for consistent analysis
    if orig_sr != SAMPLING_RATE:
        y_16k = librosa.resample(y, orig_sr=orig_sr, target_sr=SAMPLING_RATE)
    else:
        y_16k = y

    duration = float(len(y) / orig_sr)
    duration_analyzed = min(duration, float(MAX_AUDIO_SECONDS))
    y_proc = y_16k[:int(SAMPLING_RATE * duration_analyzed)]
    y_norm = librosa.util.normalize(y_proc)

    # 1. Downsampled Waveform (200 points for canvas)
    num_peaks = 200
    step = max(1, len(y_norm) // num_peaks)
    waveform_data = []
    for i in range(0, len(y_norm), step):
        chunk = y_norm[i:i + step]
        if len(chunk) > 0:
            waveform_data.append(float(np.round(np.max(np.abs(chunk)), 4)))
    waveform_data = waveform_data[:num_peaks]

    # 2. Mel Spectrogram (40 mels, converted to dB, downsampled time frames)
    mel_spec = librosa.feature.melspectrogram(
        y=y_norm,
        sr=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=40
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Downsample time frames for lightweight JSON
    if mel_db.shape[1] > max_frames:
        time_indices = np.linspace(0, mel_db.shape[1] - 1, max_frames, dtype=int)
        mel_db_down = mel_db[:, time_indices]
    else:
        mel_db_down = mel_db

    # 3. MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(
        y=y_norm,
        sr=SAMPLING_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    if mfcc.shape[1] > max_frames:
        time_indices = np.linspace(0, mfcc.shape[1] - 1, max_frames, dtype=int)
        mfcc_down = mfcc[:, time_indices]
    else:
        mfcc_down = mfcc

    # 4. Spectral Centroid and Frequency Distribution
    cent = librosa.feature.spectral_centroid(y=y_norm, sr=SAMPLING_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    avg_centroid = float(np.mean(cent))

    # FFT Magnitude spectrum
    fft_mag = np.abs(np.fft.rfft(y_norm[:min(len(y_norm), 2048)]))
    fft_freqs = np.fft.rfftfreq(min(len(y_norm), 2048), d=1.0 / SAMPLING_RATE)
    # Downsample FFT to 50 bins for chart
    fft_indices = np.linspace(0, len(fft_mag) - 1, 50, dtype=int)
    freq_data = [
        {"freq": float(round(fft_freqs[idx], 1)), "magnitude": float(round(fft_mag[idx], 3))}
        for idx in fft_indices
    ]

    return {
        "duration": round(duration, 3),
        "analyzed_duration": round(duration_analyzed, 3),
        "sampling_rate": orig_sr,
        "processed_sr": SAMPLING_RATE,
        "waveform": waveform_data,
        "mel_spectrogram": {
            "n_mels": 40,
            "frames": int(mel_db_down.shape[1]),
            "min_db": float(np.min(mel_db_down)),
            "max_db": float(np.max(mel_db_down)),
            "matrix": np.round(mel_db_down, 2).tolist()
        },
        "mfcc_heatmap": {
            "n_mfcc": N_MFCC,
            "frames": int(mfcc_down.shape[1]),
            "matrix": np.round(mfcc_down, 2).tolist()
        },
        "spectral_centroid_hz": round(avg_centroid, 1),
        "frequency_distribution": freq_data
    }

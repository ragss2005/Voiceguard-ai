from pathlib import Path
import os

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
UPLOADS_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "voiceguard.sqlite3"

# Dataset root directory configurable via environment variable ASVSPOOF_ROOT
DEFAULT_DATASET = BASE_DIR / "data" / "ASVspoof2019_LA"
DATA_ROOT = Path(os.getenv("ASVSPOOF_ROOT", str(DEFAULT_DATASET)))

# Acoustic Feature Extraction Configuration (80-D MFCC feature vector)
SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", 16000))
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 160
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", 30))
FEATURE_DIM = 80  # 20 mean + 20 std + 20 delta mean + 20 delta std

# Upload restrictions
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 25))
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}

# Privacy and retention
RETAIN_UPLOADED_AUDIO = os.getenv("RETAIN_UPLOADED_AUDIO", "false").lower() == "true"

# Model artifact paths
BEST_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
COMPARISON_PATH = ARTIFACTS_DIR / "comparison_report.json"
MODEL_LR_PATH = ARTIFACTS_DIR / "model_logistic_regression.joblib"
MODEL_SVM_PATH = ARTIFACTS_DIR / "model_svm_rbf.joblib"
MODEL_RF_PATH = ARTIFACTS_DIR / "model_random_forest.joblib"

# Backward compatibility aliases
SR = SAMPLING_RATE
HOP = HOP_LENGTH
MAX_SECONDS = MAX_AUDIO_SECONDS
MAX_UPLOAD = MAX_CONTENT_LENGTH
ALLOWED = ALLOWED_EXTENSIONS
ROOT = ROOT_DIR
BASE = BASE_DIR
ARTIFACTS = ARTIFACTS_DIR
UPLOADS = UPLOADS_DIR

# Ensure required directories exist
for folder in (ARTIFACTS_DIR, CACHE_DIR, UPLOADS_DIR, BASE_DIR / "data"):
    folder.mkdir(parents=True, exist_ok=True)

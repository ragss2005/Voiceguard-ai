# VoiceGuard AI
## Lightweight AI-Powered Synthetic Voice & Deepfake Audio Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.6-orange.svg)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/DSP-Librosa%200.11-green.svg)](https://librosa.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU--Optimized-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%20LA-blueviolet.svg)](https://www.asvspoof.org/)

**VoiceGuard AI** is a research-grade, CPU-optimized machine learning platform designed to detect synthetic speech, Text-to-Speech (TTS), and Voice Conversion (VC) deepfakes. Developed using classical machine learning algorithms and handcrafted acoustic features, the system distinguishes genuine human speech (*bonafide*) from synthetic or manipulated speech (*spoof*) with low computational overhead and sub-10ms CPU inference latency.

---

## 1. Research Overview & Problem Statement

Modern neural Text-to-Speech (TTS) models (such as VITS, ElevenLabs, XTTS, Tacotron2) and Voice Conversion (VC) systems (such as FreeVC, OpenVoice, RVC) can clone an individual's unique voice timbre from just seconds of reference audio. These capabilities create unprecedented vulnerabilities:
- Impersonation and CEO voice fraud.
- Social engineering and financial authorization bypass.
- Biometric speaker verification circumvention.

Conventional listening checks and traditional speaker identification systems fail to identify neural synthetic speech. Deep neural network (DNN) countermeasure models often require high-end NVIDIA GPUs, significant VRAM, and exhibit high inference latencies that preclude real-time deployment.

**VoiceGuard AI** addresses this gap by rigorously evaluating three classical, interpretable machine learning classifiers trained on identical 80-dimensional acoustic feature vectors:
1. **Logistic Regression** (L2 Regularized with balanced class weighting)
2. **Support Vector Machine (SVM)** (Radial Basis Function / RBF kernel with Platt scaling calibration)
3. **Random Forest** (Ensemble of balanced decision trees)

### Core Research Objective
Compare these algorithms on the official **ASVspoof 2019 Logical Access (LA)** corpus across:
- Detection performance: Accuracy, Precision, Recall, Spoof F1-Score, Genuine F1-Score.
- Equal Error Rate (EER) and Receiver Operating Characteristic (ROC-AUC).
- Computational efficiency: Model disk size, training duration, and CPU inference latency.
- Automated selection: Designate the best model based on development-set spoof-class F1-score.

---

## 2. Acoustic Feature Extraction Architecture

VoiceGuard AI enforces an identical acoustic feature pipeline across both offline training and online inference:

```
Raw Audio Signal (WAV / FLAC / MP3 / OGG / M4A / WebM)
                    │
                    ▼
          Signal Validation & Preprocessing
      (Mono downmix, 16 kHz resample, Amplitude normalization)
                    │
                    ▼
          Short-Time Fourier Transform (STFT)
            (n_fft = 512, hop_length = 160)
                    │
                    ▼
       20 Mel-Frequency Cepstral Coefficients (MFCC)
                    │
                    ├────────────────────────┐
                    ▼                        ▼
       Static MFCCs (20 coefficients)   First-Order Delta MFCCs (Δ)
          ├── Mean (20 dims)               ├── Mean (20 dims)
          └── Std Dev (20 dims)            └── Std Dev (20 dims)
                    │                        │
                    └───────────┬────────────┘
                                ▼
              80-Dimensional Acoustic Feature Vector
```

### 80-D Acoustic Vector Breakdown
- **Dims 1–20 (MFCC Mean):** Captures baseline vocal tract frequency envelope and formant positioning.
- **Dims 21–40 (MFCC Std Dev):** Measures frame-by-frame energy distribution and spectral variance.
- **Dims 41–60 (Delta-MFCC Mean):** First temporal derivative capturing phonetic transition speeds.
- **Dims 61–80 (Delta-MFCC Std Dev):** Measures transition velocity consistency, exposing unnatural synthetic temporal discontinuities.

---

## 3. Technology Stack

- **Frontend:** HTML5, CSS3 (Custom SOC Dark Theme: `#080D12`), Vanilla JavaScript (ES6+), Chart.js 4.4, Web Audio API (real-time oscilloscope), MediaRecorder API.
- **Backend:** Python 3.10+, Flask 3.1, RESTful Blueprints, Flask-CORS, Werkzeug.
- **Machine Learning & DSP:** Scikit-learn 1.6, Librosa 0.11, SciPy 1.13, SoundFile 0.13, NumPy 2.0, Joblib 1.5.
- **Database:** SQLite 3 (Auto-migrating schema with thread-safe connection pooling).
- **Deployment:** Gunicorn WSGI server, Docker, Docker Compose.

---

## 4. Project Directory Layout

```
voiceguard-ai/
│
├── backend/
│   ├── app.py                     # Flask application factory & static router
│   ├── config.py                  # System paths, DSP constants, upload limits
│   ├── database.py                # SQLite schema, migrations & CRUD operations
│   ├── voiceguard.sqlite3         # SQLite database file
│   │
│   ├── routes/
│   │   ├── analysis.py            # POST /api/analyze, /api/audio/visualize
│   │   ├── analytics.py           # GET /api/stats, /api/analytics
│   │   ├── history.py             # GET/DELETE /api/history, CSV export
│   │   ├── models.py              # GET/POST /api/models, /api/models/select
│   │   └── dataset.py             # GET /api/dataset/info, POST /api/training/start
│   │
│   ├── ml/
│   │   ├── features.py            # 80-D MFCC extractor & spectrogram generator
│   │   ├── dataset_loader.py      # ASVspoof protocol parser, indexer, feature cache
│   │   ├── train.py               # 3-model benchmark & best model selector
│   │   ├── metrics.py             # EER calculation, ROC curve, confusion matrices
│   │   └── predict.py             # Model inference, score calibration & disclaimers
│   │
│   ├── services/
│   │   ├── audio_service.py       # Upload validation, safe storage, file cleanup
│   │   ├── model_service.py       # Active model management & comparison loader
│   │   └── training_service.py    # Background threaded trainer with live logs
│   │
│   └── artifacts/                 # Serialized models, caches, and metric reports
│       ├── best_model.joblib
│       ├── model_logistic_regression.joblib
│       ├── model_svm_rbf.joblib
│       ├── model_random_forest.joblib
│       ├── metrics.json
│       ├── comparison_report.json
│       └── cache/                 # Compressed .npz feature caches
│
├── frontend/
│   ├── index.html                 # Single-page SOC application interface
│   ├── css/
│   │   └── style.css              # Dark forensic stylesheet (#080D12 theme)
│   └── js/
│       └── app.js                 # Reactive frontend, mic recording, Chart.js
│
├── tests/
│   ├── test_features.py           # Audio validation & 80-D MFCC feature tests
│   ├── test_models.py             # ML pipeline, EER, and calibration tests
│   ├── test_api.py                # REST API endpoints & database integration tests
│   └── test_dataset.py            # Protocol parser & feature cache tests
│
├── scripts/
│   └── generate_demo_benchmarks.py# Quick demonstration benchmark generator
│
├── requirements.txt               # Pinned dependencies
├── .env.example                   # Environment variable template
├── Dockerfile                     # CPU-optimized multi-stage Docker build
├── docker-compose.yml             # Container orchestration
├── gunicorn.conf.py               # Production WSGI server configuration
└── README.md                      # Comprehensive technical documentation
```

---

## 5. Installation & Local Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- `libsndfile1` (standard on macOS/Windows; on Debian/Ubuntu: `sudo apt-get install -y libsndfile1 ffmpeg`)

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/voiceguard-ai.git
cd voiceguard-ai

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate initial demonstration benchmark (Optional if ASVspoof not yet downloaded)
python3 scripts/generate_demo_benchmarks.py

# 5. Start the VoiceGuard AI platform
python3 -m backend.app
```

Navigate to **http://127.0.0.1:5000** in your browser.

---

## 6. ASVspoof 2019 Dataset Integration

VoiceGuard AI is engineered to integrate with the official **ASVspoof 2019 Logical Access (LA)** database.

### Download Instructions
1. Visit the official Edinburgh DataShare portal:
   [https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45)
2. Download `LA.zip` (~25 GB).
3. Unzip into `backend/data/ASVspoof2019_LA` or an external drive.
4. Verify the directory contains:
   - `ASVspoof2019_LA_train/flac/`
   - `ASVspoof2019_LA_dev/flac/`
   - `ASVspoof2019_LA_eval/flac/`
   - `ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt`
   - `ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt`

### Dataset Path Configuration
Configure the path using the `ASVSPOOF_ROOT` environment variable without modifying code:

```bash
# macOS / Linux
export ASVSPOOF_ROOT="/Volumes/ExternalDrive/ASVspoof2019_LA"

# Windows PowerShell
$env:ASVSPOOF_ROOT="D:\Datasets\ASVspoof2019_LA"
```

---

## 7. Model Training & Benchmarking

### Method A: Web UI (Model Management Tab)
1. Open http://127.0.0.1:5000 and click **Model Management** in the left sidebar.
2. (Optional) Set `Max Train Samples` and `Max Dev Samples` for rapid evaluation.
3. Click **▶ Start Background Training**.
4. The background worker extracts features, trains all three algorithms, computes EER/ROC, selects the winning model, and updates the dashboard without blocking web requests.

### Method B: Command-Line Interface (CLI)
```bash
# Run training on all available ASVspoof 2019 LA samples:
python3 -m backend.ml.train

# Smoke test on a subset of samples:
python3 -m backend.ml.train --max-train 500 --max-dev 200

# Disable disk feature cache to force fresh extraction:
python3 -m backend.ml.train --no-cache
```

---

## 8. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health, model readiness, active classifier |
| `GET` | `/api/stats` | Dashboard statistics, classification counts, latency |
| `POST` | `/api/analyze` | Multipart upload audio analysis (`audio`, optional `model`) |
| `POST` | `/api/audio/visualize` | Waveform, Mel-Spectrogram, and MFCC heatmap payload |
| `GET` | `/api/history` | Paginated audit log (`limit`, `offset`, `label`, `q`) |
| `GET` | `/api/history/<id>` | Full record metadata inspection |
| `DELETE` | `/api/history/<id>` | Deletes individual detection record |
| `DELETE` | `/api/history` | Clears all detection records |
| `GET` | `/api/history/export` | Downloads detection log as CSV |
| `GET` | `/api/models` | Lists all models, metrics, active status |
| `GET` | `/api/models/compare` | JSON comparison report of the 3 algorithms |
| `POST` | `/api/models/select` | Switches the active model (`model_name`) |
| `GET` | `/api/dataset/info` | ASVspoof path status, partition file counts |
| `POST` | `/api/training/start` | Initiates background training pipeline |
| `GET` | `/api/training/status`| Live training progress, logs, and completion status |
| `GET` | `/api/analytics` | Research analytics, ROC points, confusion matrices |

---

## 9. Testing & Quality Assurance

VoiceGuard AI includes a suite of 20 automated unit and integration tests covering feature extraction, audio validation, ML pipelines, metrics, REST APIs, and database persistence.

To execute the test suite:
```bash
# Using Python's built-in test runner:
python3 -m unittest discover tests

# Or using pytest (if installed):
pytest tests/ -v
```

### Verified Test Matrix
- `tests/test_features.py`: Exact 80-D MFCC feature dimension, stereo downmix, 44.1kHz->16kHz resampling, empty audio rejection, silence rejection, short clip rejection, spectrogram generation.
- `tests/test_models.py`: EER calculation, ROC curve interpolation, Logistic Regression fitting, SVM RBF fitting, Random Forest fitting, score calibration.
- `tests/test_api.py`: `/api/health`, `/api/stats`, `/api/models`, `/api/dataset/info`, `/api/analyze` input validation, multipart form upload, database insertion, CSV export.
- `tests/test_dataset.py`: ASVspoof protocol parsing, audio indexing, `.npz` feature cache save and load, unconfigured directory diagnostics.

---

## 10. Production Deployment

### Docker Deployment
```bash
# Build and run container
docker-compose up -d --build

# Inspect logs
docker-compose logs -f
```

### Production Gunicorn Server
```bash
gunicorn --config gunicorn.conf.py backend.app:app
```

---

## 11. Ethical Considerations & Limitations

1. **Research Screening Tool:** VoiceGuard AI is designed as a forensic acoustic screening tool, not a certified biometric authentication system.
2. **Confidence Scores:** Confidence percentages represent statistical correlation with training acoustic distributions; they do not establish legal proof of speaker identity or deception.
3. **Generalization:** Unseen neural vocoders, severe telephone band-pass filtering (G.711 / AMR), heavy background noise, or cross-language code-switching may degrade classification accuracy.
4. **Privacy:** The application automatically purges temporary audio files after feature extraction. Audio retention can be toggled in `backend/config.py`.

---

## 12. Academic Citation

If you use VoiceGuard AI in academic work, engineering theses, or research publications, please cite:

```bibtex
@misc{voiceguard_ai_2026,
  author = {VoiceGuard AI Research Team},
  title = {VoiceGuard AI: Lightweight Machine Learning-Based Synthetic Voice and Audio Deepfake Detection System},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/your-username/voiceguard-ai}}
}
```

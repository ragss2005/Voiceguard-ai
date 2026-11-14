# VoiceGuard AI - Lightweight CPU-Optimized Synthetic Voice Detection
FROM python:3.10-slim

# Install system audio libraries and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY gunicorn.conf.py .

# Environment configuration
ENV FLASK_ENV=production \
    FLASK_PORT=5000 \
    FLASK_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# Run with Gunicorn WSGI server
CMD ["gunicorn", "--config", "gunicorn.conf.py", "backend.app:app"]

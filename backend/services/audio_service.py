import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from backend.config import UPLOADS_DIR, ALLOWED_EXTENSIONS, RETAIN_UPLOADED_AUDIO
from backend.ml.features import validate_audio, generate_spectrogram, AudioValidationError

class AudioService:
    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        if not filename or "." not in filename:
            return False
        ext = Path(filename).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    @staticmethod
    def save_upload(file_storage: FileStorage) -> Tuple[Path, str]:
        """
        Saves uploaded file to a unique path in UPLOADS_DIR.
        Returns: (saved_path, original_filename)
        """
        original_name = file_storage.filename or "upload.wav"
        safe_name = secure_filename(original_name) or "audio.wav"
        ext = Path(safe_name).suffix.lower()
        if not ext:
            ext = ".wav"
            safe_name += ext

        unique_filename = f"{uuid.uuid4().hex[:16]}_{safe_name}"
        save_path = UPLOADS_DIR / unique_filename
        file_storage.save(save_path)
        return save_path, original_name

    @staticmethod
    def cleanup_file(file_path: Path):
        """Safely removes temporary audio file unless audio retention is enabled."""
        if not RETAIN_UPLOADED_AUDIO and file_path and file_path.exists():
            try:
                file_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Failed to delete temporary audio file {file_path}: {e}")

    @staticmethod
    def get_audio_visuals(file_path: Path) -> Dict[str, Any]:
        """Generates audio visualization payload (waveform, spectrogram, mfcc, frequency distribution)."""
        return generate_spectrogram(file_path)

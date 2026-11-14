import io
import unittest
import numpy as np
import soundfile as sf
from backend.config import FEATURE_DIM, SAMPLING_RATE
from backend.ml.features import (
    extract_features_from_waveform,
    extract_features,
    validate_audio,
    generate_spectrogram,
    AudioValidationError
)

class TestAudioFeatures(unittest.TestCase):
    def setUp(self):
        # Generate 1.0 second test tone at 16,000 Hz
        t = np.linspace(0, 1.0, SAMPLING_RATE, endpoint=False)
        self.sine_16k = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    def test_waveform_features_dimension(self):
        """Feature vector must be exactly 80 dimensions."""
        vec = extract_features_from_waveform(self.sine_16k, SAMPLING_RATE)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_stereo_to_mono_conversion(self):
        """Stereo 2-channel audio must be converted to mono and produce 80-D vector."""
        stereo = np.stack([self.sine_16k, self.sine_16k], axis=0) # (2, samples)
        vec = extract_features_from_waveform(stereo, SAMPLING_RATE)
        self.assertEqual(vec.shape, (FEATURE_DIM,))

    def test_resampling_from_44k(self):
        """Audio sampled at 44.1 kHz must be resampled and produce 80-D vector."""
        sr_44k = 44100
        t = np.linspace(0, 1.0, sr_44k, endpoint=False)
        sine_44k = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        vec = extract_features_from_waveform(sine_44k, sr_44k)
        self.assertEqual(vec.shape, (FEATURE_DIM,))

    def test_silent_audio_rejected(self):
        """Audio containing only zeroes or near-zero amplitude must raise AudioValidationError."""
        silent = np.zeros(SAMPLING_RATE, dtype=np.float32)
        with self.assertRaises(AudioValidationError):
            validate_audio(io.BytesIO(self._to_wav_bytes(silent, SAMPLING_RATE)))

    def test_empty_audio_rejected(self):
        """Completely empty audio data must raise AudioValidationError."""
        empty = np.array([], dtype=np.float32)
        with self.assertRaises(AudioValidationError):
            validate_audio(io.BytesIO(self._to_wav_bytes(empty, SAMPLING_RATE)))

    def test_too_short_audio_rejected(self):
        """Audio shorter than 0.1s must be rejected."""
        short_audio = self.sine_16k[:int(SAMPLING_RATE * 0.05)]
        with self.assertRaises(AudioValidationError):
            validate_audio(io.BytesIO(self._to_wav_bytes(short_audio, SAMPLING_RATE)))

    def test_generate_spectrogram_structure(self):
        """Spectrogram generator must output valid JSON-serializable visual data."""
        wav_bytes = self._to_wav_bytes(self.sine_16k, SAMPLING_RATE)
        data = generate_spectrogram(io.BytesIO(wav_bytes))
        self.assertIn("waveform", data)
        self.assertIn("mel_spectrogram", data)
        self.assertIn("mfcc_heatmap", data)
        self.assertIn("frequency_distribution", data)
        self.assertGreater(len(data["waveform"]), 0)
        self.assertEqual(data["mel_spectrogram"]["n_mels"], 40)
        self.assertEqual(data["mfcc_heatmap"]["n_mfcc"], 20)

    def _to_wav_bytes(self, samples, sr):
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)
        return buf.read()

if __name__ == "__main__":
    unittest.main()

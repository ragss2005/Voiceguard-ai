import io
import unittest
import numpy as np
import soundfile as sf
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app import create_app
from backend.config import ARTIFACTS_DIR, BEST_MODEL_PATH, SAMPLING_RATE, FEATURE_DIM
from backend.database import init_db, clear_all_history

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Create a small dummy model bundle so analyze endpoint can run during software test
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        dummy_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ])
        # Fit on small 80-D synthetic fixture
        X = np.random.randn(20, FEATURE_DIM).astype(np.float32)
        y = np.array([0]*10 + [1]*10)
        dummy_clf.fit(X, y)

        joblib.dump({
            "model": dummy_clf,
            "model_name": "Test Logistic Regression",
            "algorithm": "Logistic Regression Test Fixture",
            "feature_dim": FEATURE_DIM
        }, BEST_MODEL_PATH)

    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_ready"])

    def test_stats_endpoint(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total", data)
        self.assertIn("synthetic", data)
        self.assertIn("genuine", data)

    def test_dataset_info_endpoint(self):
        res = self.client.get("/api/dataset/info")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("root_path", data)
        self.assertIn("partitions", data)

    def test_models_list_endpoint(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("models", data)
        self.assertIn("system", data)

    def test_analyze_no_file(self):
        res = self.client.post("/api/analyze")
        self.assertEqual(res.status_code, 400)

    def test_analyze_invalid_extension(self):
        data = {"audio": (io.BytesIO(b"fake txt content"), "test.txt")}
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 415)

    def test_analyze_valid_audio(self):
        # Generate 1.0 second sine wave
        t = np.linspace(0, 1.0, SAMPLING_RATE, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, sine, SAMPLING_RATE, format="WAV")
        buf.seek(0)

        data = {"audio": (buf, "sample_test.wav")}
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        self.assertIn("id", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertIn("synthetic_score", result)
        self.assertIn("genuine_score", result)
        self.assertIn("latency_ms", result)
        self.assertIn("visuals", result)

        # Verify history record was created
        hist_res = self.client.get("/api/history")
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()
        self.assertGreater(hist_data["total"], 0)

        # Test single history item retrieval
        single_res = self.client.get(f"/api/history/{result['id']}")
        self.assertEqual(single_res.status_code, 200)
        self.assertEqual(single_res.get_json()["id"], result["id"])

        # Test CSV export
        export_res = self.client.get("/api/history/export")
        self.assertEqual(export_res.status_code, 200)
        self.assertIn(b"Analysis ID", export_res.data)

        # Test delete single record
        del_res = self.client.delete(f"/api/history/{result['id']}")
        self.assertEqual(del_res.status_code, 200)

    def test_visualize_endpoint(self):
        t = np.linspace(0, 0.5, SAMPLING_RATE, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, sine, SAMPLING_RATE, format="WAV")
        buf.seek(0)

        res = self.client.post("/api/audio/visualize", data={"audio": (buf, "tone.wav")}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("waveform", data)
        self.assertIn("mel_spectrogram", data)
        self.assertIn("mfcc_heatmap", data)

if __name__ == "__main__":
    unittest.main()

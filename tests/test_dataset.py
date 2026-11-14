import tempfile
import unittest
from pathlib import Path
import numpy as np

from backend.config import FEATURE_DIM
from backend.ml.dataset_loader import (
    parse_protocol,
    save_cached_features,
    load_cached_features,
    get_dataset_diagnostics,
    build_audio_index
)

class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_protocol_sample(self):
        """Protocol parsing must extract audio ID, label int, and attack type."""
        proto_file = self.root / "sample_protocol.txt"
        proto_file.write_text(
            "LA_0001 LA_T_1000137 - - bonafide\n"
            "LA_0002 LA_T_1000138 - A01 spoof\n"
            "LA_0003 LA_T_1000139 - A02 spoof\n"
        )

        rows = parse_protocol(proto_file)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ("LA_T_1000137.flac", 0, "-"))
        self.assertEqual(rows[1], ("LA_T_1000138.flac", 1, "A01"))
        self.assertEqual(rows[2], ("LA_T_1000139.flac", 1, "A02"))

    def test_cache_save_and_load(self):
        """Cached features must save as compressed .npz and load with exact dimensions."""
        cache_file = self.root / "cache_test.npz"
        X = np.random.randn(10, FEATURE_DIM).astype(np.float32)
        y = np.array([0]*5 + [1]*5, dtype=int)
        filenames = [f"audio_{i}.flac" for i in range(10)]

        save_cached_features(cache_file, X, y, filenames)
        self.assertTrue(cache_file.exists())

        loaded = load_cached_features(cache_file)
        self.assertIsNotNone(loaded)
        X_loaded, y_loaded, fnames_loaded = loaded
        np.testing.assert_array_almost_equal(X, X_loaded)
        np.testing.assert_array_equal(y, y_loaded)
        self.assertEqual(filenames, fnames_loaded)

    def test_dataset_diagnostics_unconfigured(self):
        """Unconfigured or non-existent path must report clean diagnostic object without crashing."""
        diag = get_dataset_diagnostics(self.root / "nonexistent")
        self.assertFalse(diag["root_exists"])
        self.assertFalse(diag["ready_for_training"])
        self.assertIn("missing_components", diag)

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import joblib
from backend.config import FEATURE_DIM
from backend.ml.metrics import compute_eer, evaluate_model_performance
from backend.ml.train import get_model_definitions
from backend.ml.predict import predict_features

class TestMachineLearningPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create synthetic fixture dataset (100 samples, 80 features) for unit testing code paths
        self.n_samples = 100
        # Class 0 (bonafide): mean -0.5
        X_0 = np.random.normal(loc=-0.5, scale=1.0, size=(self.n_samples // 2, FEATURE_DIM))
        y_0 = np.zeros(self.n_samples // 2, dtype=int)
        
        # Class 1 (spoof): mean +0.5
        X_1 = np.random.normal(loc=0.5, scale=1.0, size=(self.n_samples // 2, FEATURE_DIM))
        y_1 = np.ones(self.n_samples // 2, dtype=int)

        self.X_train = np.vstack([X_0[:30], X_1[:30]]).astype(np.float32)
        self.y_train = np.concatenate([y_0[:30], y_1[:30]])

        self.X_val = np.vstack([X_0[30:], X_1[30:]]).astype(np.float32)
        self.y_val = np.concatenate([y_0[30:], y_1[30:]])

    def test_eer_computation(self):
        """Equal Error Rate calculation should return float between 0.0 and 1.0."""
        # Perfect scores
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        eer, thresh = compute_eer(y_true, y_scores)
        self.assertAlmostEqual(eer, 0.0, places=2)

        # Inverted scores
        y_scores_bad = np.array([0.9, 0.8, 0.2, 0.1])
        eer_bad, _ = compute_eer(y_true, y_scores_bad)
        self.assertGreater(eer_bad, 0.4)

    def test_train_all_three_algorithms(self):
        """Logistic Regression, SVM, and Random Forest must all fit and evaluate cleanly."""
        models = get_model_definitions()
        self.assertEqual(len(models), 3)

        for name, cfg in models.items():
            pipeline = cfg["pipeline"]
            pipeline.fit(self.X_train, self.y_train)

            # Evaluate metrics
            metrics = evaluate_model_performance(pipeline, self.X_val, self.y_val, model_name=name)
            
            self.assertIn("accuracy", metrics)
            self.assertIn("f1_spoof", metrics)
            self.assertIn("precision_spoof", metrics)
            self.assertIn("recall_spoof", metrics)
            self.assertIn("eer", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("avg_latency_ms", metrics)

            self.assertGreaterEqual(metrics["accuracy"], 0.0)
            self.assertLessEqual(metrics["accuracy"], 1.0)
            self.assertGreaterEqual(metrics["eer"], 0.0)
            self.assertLessEqual(metrics["eer"], 1.0)

            # Test predict_features
            bundle = {
                "model": pipeline,
                "model_name": name,
                "algorithm": cfg["algorithm"],
                "feature_dim": FEATURE_DIM
            }
            sample_feat = self.X_val[0]
            pred_out = predict_features(sample_feat, bundle)

            self.assertIn(pred_out["label"], ("genuine", "synthetic"))
            self.assertIn("confidence", pred_out)
            self.assertIn("synthetic_score", pred_out)
            self.assertIn("genuine_score", pred_out)
            self.assertIn("explanation", pred_out)
            self.assertIn("disclaimer", pred_out)

if __name__ == "__main__":
    unittest.main()

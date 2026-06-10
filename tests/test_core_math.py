import unittest

import numpy as np

from src.evaluation.betting import expected_value
from src.evaluation.model_ensemble import _apply_calibration_matrix
from src.models.poisson_model import derive_markets, dixon_coles_adjust, score_matrix


class CoreMathTest(unittest.TestCase):
    def test_expected_value(self):
        self.assertAlmostEqual(expected_value(0.55, 2.0), 0.10)
        self.assertAlmostEqual(expected_value(0.40, 2.0), -0.20)

    def test_score_matrix_is_valid_distribution(self):
        matrix = score_matrix(1.4, 1.1)
        self.assertEqual(matrix.shape, (9, 9))
        self.assertTrue(np.all(matrix >= 0))
        self.assertLessEqual(matrix.sum(), 1.0)
        self.assertGreater(matrix.sum(), 0.99)

    def test_dixon_coles_adjustment_is_valid_distribution(self):
        matrix = score_matrix(1.4, 1.1)
        adjusted = dixon_coles_adjust(matrix, 1.4, 1.1, rho=-0.08)
        self.assertEqual(adjusted.shape, matrix.shape)
        self.assertTrue(np.all(adjusted >= 0))
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=8)

    def test_negative_dixon_coles_rho_increases_low_score_draws(self):
        independent = score_matrix(1.4, 1.1)
        adjusted = score_matrix(1.4, 1.1, rho=-0.08)
        self.assertGreater(adjusted[0, 0], independent[0, 0])
        self.assertGreater(adjusted[1, 1], independent[1, 1])

    def test_derived_result_probabilities_are_coherent(self):
        matrix = score_matrix(1.4, 1.1)
        markets = derive_markets(matrix)
        total = markets["prob_H"] + markets["prob_D"] + markets["prob_A"]
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertGreaterEqual(markets["prob_over25"], 0.0)
        self.assertLessEqual(markets["prob_over25"], 1.0)
        self.assertGreaterEqual(markets["prob_btts"], 0.0)
        self.assertLessEqual(markets["prob_btts"], 1.0)

    def test_ensemble_calibration_preserves_probability_distribution(self):
        probs = np.array([
            [0.50, 0.25, 0.25],
            [0.20, 0.30, 0.50],
        ])
        calibration = {
            "calibrators": {
                "H": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.05, 0.90]},
                "D": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.10, 0.40]},
                "A": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.05, 0.90]},
            }
        }

        calibrated = _apply_calibration_matrix(probs, calibration)

        self.assertEqual(calibrated.shape, probs.shape)
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertTrue(np.all(calibrated <= 1.0))
        np.testing.assert_allclose(calibrated.sum(axis=1), np.ones(len(probs)))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from src.evaluation.betting import expected_value
from src.models.poisson_model import derive_markets, score_matrix


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

    def test_derived_result_probabilities_are_coherent(self):
        matrix = score_matrix(1.4, 1.1)
        markets = derive_markets(matrix)
        total = markets["prob_H"] + markets["prob_D"] + markets["prob_A"]
        self.assertAlmostEqual(total, 1.0, places=3)
        self.assertGreaterEqual(markets["prob_over25"], 0.0)
        self.assertLessEqual(markets["prob_over25"], 1.0)
        self.assertGreaterEqual(markets["prob_btts"], 0.0)
        self.assertLessEqual(markets["prob_btts"], 1.0)


if __name__ == "__main__":
    unittest.main()

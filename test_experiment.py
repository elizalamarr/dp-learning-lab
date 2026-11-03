"""Check preprocessing and budget accounting, not a proof of privacy."""

import unittest

import numpy as np

from experiment import clip_rows, evaluate


class ExperimentTests(unittest.TestCase):
    def test_clipping_bounds_rows_and_preserves_small_vectors(self):
        x = np.array([[0., 0.], [3., 4.], [-6., 8.], [.1, .2]])
        clipped = clip_rows(x)
        self.assertTrue(np.all(np.linalg.norm(clipped, axis=1) <= 1 + 1e-12))
        np.testing.assert_allclose(clipped, [[0, 0], [.6, .8], [-.6, .8], [.1, .2]])
        np.testing.assert_array_equal(x, [[0, 0], [3, 4], [-6, 8], [.1, .2]])

    def test_budget_counts_every_private_fit(self):
        baseline, rows, budget = evaluate(epsilons=(1.0, 3.0), repeats=2)
        self.assertEqual([(row["epsilon"], row["repeat"]) for row in rows],
                         [(1.0, 0), (1.0, 1), (3.0, 0), (3.0, 1)])
        self.assertAlmostEqual(budget[0], 8.0)
        self.assertEqual(budget[1], 0.0)
        self.assertTrue(0 <= baseline <= 1)
        self.assertTrue(all(0 <= row["accuracy"] <= 1 for row in rows))

    def test_budget_is_local_to_each_evaluation(self):
        for _ in range(2):
            _, rows, budget = evaluate(epsilons=(0.5,), repeats=1)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(budget[0], 0.5)
            self.assertEqual(budget[1], 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from automation.train_guidance_selector_bc import augment_mirrors, metrics


class GuidanceBCTests(unittest.TestCase):
    def test_mirror_augmentation_quadruples_samples(self):
        x = np.zeros((3, 45), dtype=np.float32)
        y = np.array([0, 1, 3], dtype=np.int64)
        augmented_x, augmented_y = augment_mirrors(x, y)
        self.assertEqual(augmented_x.shape, (12, 45))
        self.assertEqual(augmented_y.shape, (12,))
        self.assertEqual(int(np.sum(augmented_y == 2)), 2)
        self.assertEqual(int(np.sum(augmented_y == 4)), 2)

    def test_metrics_report_default_recall_and_nondefault_precision(self):
        y = np.array([0, 0, 1, 2], dtype=np.int64)
        probabilities = np.eye(9, dtype=np.float32)[y]
        result = metrics(y, probabilities)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["bt_default_recall"], 1.0)
        self.assertEqual(result["nondefault_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()


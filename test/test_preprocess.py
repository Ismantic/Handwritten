from __future__ import annotations

import unittest

import numpy as np

try:
    from src.preprocess import preprocess
except OSError:
    preprocess = None

try:
    from src.normalize import normalize
except ModuleNotFoundError:
    normalize = None


class PreprocessParityTest(unittest.TestCase):
    @unittest.skipIf(preprocess is None, "run `make -C src preprocess` first")
    def test_c_output_contract(self) -> None:
        image = np.full((103, 87), 255, dtype=np.uint8)
        image[12:91, 39:47] = 0
        actual = preprocess(image)
        self.assertEqual(actual.shape, (1, 64, 64))
        self.assertEqual(actual.dtype, np.float32)
        self.assertGreater(float(actual.max()), 0.0)
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)

    @unittest.skipIf(preprocess is None, "run `make -C src preprocess` first")
    @unittest.skipIf(normalize is None, "install Pillow to compare Python reference")
    def test_c_matches_python_reference(self) -> None:
        image = np.full((103, 87), 255, dtype=np.uint8)
        image[12:91, 39:47] = 0
        expected = (255 - normalize(image)).astype(np.float32) / 255.0
        actual = preprocess(image)[0]
        np.testing.assert_allclose(actual, expected, atol=1.0 / 255.0)


if __name__ == "__main__":
    unittest.main()

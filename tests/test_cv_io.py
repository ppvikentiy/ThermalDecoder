"""BMP read/write via cv_io (Unicode-safe path I/O)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from thermal_decoder import cv_io
from thermal_decoder.cv_io import imread_bgr, imwrite


class TestCvIoBmp(unittest.TestCase):
    def test_roundtrip_bgr_bmp(self) -> None:
        img = np.zeros((24, 32, 3), dtype=np.uint8)
        img[:, :] = (40, 100, 200)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.bmp"
            self.assertTrue(imwrite(p, img))
            back = imread_bgr(p)
            self.assertIsNotNone(back)
            assert back is not None
            self.assertEqual(back.shape, img.shape)
            np.testing.assert_array_equal(back, img)

    def test_grayscale_bmp_becomes_bgr(self) -> None:
        g = np.full((16, 20), 77, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "gray.bmp"
            self.assertTrue(imwrite(p, g))
            back = imread_bgr(p)
            self.assertIsNotNone(back)
            assert back is not None
            self.assertEqual(back.shape, (16, 20, 3))
            np.testing.assert_array_equal(back[:, :, 0], g)
            np.testing.assert_array_equal(back[:, :, 1], g)
            np.testing.assert_array_equal(back[:, :, 2], g)

    def test_uint16_grayscale_decodes_to_bgr(self) -> None:
        h, w = 8, 10
        u16 = (np.random.rand(h, w) * 1000).astype(np.uint16) + 200
        bgr = cv_io._numpy_to_bgr_u8(u16)  # noqa: SLF001
        self.assertIsNotNone(bgr)
        assert bgr is not None
        self.assertEqual(bgr.shape, (h, w, 3))

    def test_uint16_bgr_3ch(self) -> None:
        h, w = 4, 5
        u16 = (np.random.rand(h, w, 3) * 500).astype(np.uint16) + 100
        bgr = cv_io._numpy_to_bgr_u8(u16)  # noqa: SLF001
        self.assertIsNotNone(bgr)
        assert bgr is not None
        self.assertEqual(bgr.shape, (h, w, 3))
        self.assertEqual(bgr.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for scale gradient validation."""

from __future__ import annotations

import unittest

import numpy as np

from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder.scale_detector import ScaleDetector


class TestScaleGradient(unittest.TestCase):
    def test_smooth_gradient_passes(self) -> None:
        n = 64
        ramp = np.zeros((n, 1, 3), dtype=np.uint8)
        for i in range(n):
            v = int(255 * i / (n - 1))
            ramp[i, 0] = (v, 128, 255 - v)
        colors = ramp.reshape(n, 3)
        ScaleDetector().validate_scale_gradient(colors, strict=True)

    def test_broken_gradient_raises(self) -> None:
        n = 40
        colors = np.zeros((n, 3), dtype=np.uint8)
        colors[:, 1] = 128
        colors[:, 2] = 200
        colors[:20, 0] = np.linspace(50, 200, 20).astype(np.uint8)
        colors[20:, 0] = np.linspace(20, 250, 20).astype(np.uint8)
        with self.assertRaises(System_OCV_Vis_Temp_Error):
            ScaleDetector().validate_scale_gradient(colors, strict=True)

    def test_sharp_end_caps_allowed(self) -> None:
        """Thermal palettes often clip to white/black at the extremes (steep end steps)."""
        n = 50
        colors = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            v = int(50 + 200 * i / (n - 1))
            colors[i] = (v, 100, 255 - v)
        colors[0] = (255, 255, 255)
        colors[-1] = (0, 0, 30)
        ScaleDetector().validate_scale_gradient(colors, strict=True)


class TestGetScaleColors(unittest.TestCase):
    def test_median_rows(self) -> None:
        roi = np.zeros((4, 3, 3), dtype=np.uint8)
        roi[:, 0, :] = (10, 20, 30)
        roi[:, 1, :] = (50, 60, 70)
        roi[:, 2, :] = (90, 100, 110)
        out = ScaleDetector().get_scale_colors(roi)
        self.assertEqual(out.shape, (4, 3))
        np.testing.assert_array_equal(out[0], (50, 60, 70))

    def test_wide_roi_uses_right_strip_for_gradient(self) -> None:
        """Noise on the left must not break validation of a smooth bar on the right."""
        h, w = 48, 40
        roi = np.random.default_rng(0).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        narrow = max(4, w // 4)
        for i in range(h):
            v = int(255 * i / (h - 1))
            roi[i, -narrow:, :] = (v, 128, 255 - v)
        colors = ScaleDetector().get_scale_colors(roi)
        ScaleDetector().validate_scale_gradient(colors, strict=True)


if __name__ == "__main__":
    unittest.main()

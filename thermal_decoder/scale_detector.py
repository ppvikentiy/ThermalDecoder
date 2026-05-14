"""Vertical temperature-scale detection and validation."""

from __future__ import annotations

import numpy as np
import cv2

from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder import constants as C


class ScaleDetector:
    """Finds the vertical color bar on the right and validates its gradient."""

    def detect_scale(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Find vertical gradient strip on the right.
        Returns (x, y, w, h) in full-image coordinates, or None.
        """
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        if h < 8 or w < 8:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_x0 = int(w * 0.70)
        strip = gray[:, roi_x0:]
        hs, ws = strip.shape
        if ws < 5:
            return None

        scores = np.empty(ws, dtype=np.float32)
        for j in range(ws):
            col = strip[:, j].astype(np.float32)
            scores[j] = float(np.sum(np.abs(np.diff(col))))

        win = min(48, max(8, ws // 5))
        if win > ws:
            return None
        kernel = np.ones(win, dtype=np.float32) / win
        smoothed = np.convolve(scores, kernel, mode="valid")
        if smoothed.size == 0:
            return None
        j0 = int(np.argmax(smoothed))
        x = roi_x0 + j0
        bar_w = min(win, w - x)
        if bar_w < 4:
            return None
        y = 0
        bar_h = h
        return (x, y, bar_w, bar_h)

    def get_scale_colors(self, roi: np.ndarray) -> np.ndarray:
        """
        Extract BGR colors top-to-bottom (median across width per row).

        For wide ROIs, only the right quarter of columns is used (min 4 px).
        The color bar is usually on the right; medians over the full width
        mix in labels/background and break gradient validation.
        """
        if roi is None or roi.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError("ROI must be a BGR image")
        _h, rw = roi.shape[:2]
        if rw >= 8:
            narrow = max(4, min(rw, rw // 4))
            strip = roi[:, -narrow:, :]
        else:
            strip = roi
        row_med = np.median(strip.astype(np.float32), axis=1)
        return np.clip(row_med, 0, 255).astype(np.uint8)

    def validate_scale_gradient(
        self,
        colors: np.ndarray,
        *,
        strict: bool = True,
    ) -> None:
        """
        Raise System_OCV_Vis_Temp_Error if the palette has a discontinuous jump.
        """
        if colors.shape[0] < 3:
            raise System_OCV_Vis_Temp_Error(
                detail=(
                    "В области шкалы слишком мало строк (нужно не меньше 3). "
                    "Увеличьте высоту прямоугольника по цветной полоске."
                )
            )

        lab = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(
            np.float32
        ).reshape(-1, 3)
        d = np.sqrt(np.sum(np.diff(lab, axis=0) ** 2, axis=1))
        positive = d[d > 1e-3]
        if positive.size == 0:
            raise System_OCV_Vis_Temp_Error(
                detail=(
                    "По вертикали почти нет изменения цвета. Убедитесь, что в "
                    "прямоугольнике только цветная шкала, а не однотонный фон."
                )
            )
        med = float(np.median(positive))
        mult = (
            C.gradient_strict_multiplier
            if strict
            else C.gradient_soft_multiplier
        )
        floor = (
            C.gradient_delta_e_floor_strict
            if strict
            else C.gradient_delta_e_floor_soft
        )
        thresh = max(med * mult, floor)
        # Термокамеры часто дают очень резкий переход у min/max (белый наконечник,
        # чёрное основание). Это 1–3 больших Δ только у краёв последовательности.
        # Проверяем непрерывность по «сердцевине», иначе ложное срабатывание при
        # корректно выделенной полоске.
        nd = int(d.shape[0])
        if nd >= 6:
            trim = max(1, min(24, nd // 7))
            d_chk = d[trim : nd - trim] if nd > 2 * trim else d
        else:
            d_chk = d
        if d_chk.size == 0:
            d_chk = d

        if np.any(d_chk > thresh):
            mode = "строгая" if strict else "мягкая"
            raise System_OCV_Vis_Temp_Error(
                detail=(
                    "Резкий скачок цвета в середине шкалы: часто из‑за цифр, рамки, "
                    "не той области или ступенчатой палитры. Если полоска одна и без "
                    "подписей, попробуйте режим «Мягкий» или чуть сузьте/сместите "
                    f"прямоугольник (проверка градиента: {mode})."
                )
            )

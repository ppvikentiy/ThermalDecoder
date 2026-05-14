"""Main decoder: load BMP, map colors to temperature, export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from thermal_decoder import constants as C
from thermal_decoder.cv_io import imread_bgr, imwrite as imwrite_image
from thermal_decoder.exceptions import System_OCV_Vis_Temp_Error
from thermal_decoder.io_export import build_result_overlay_file, save_temperature_csv
from thermal_decoder.scale_detector import ScaleDetector


class ThermalDecoder:
    """Loads thermal BMPs, calibrates from scale ROI, computes temperature grids."""

    def __init__(self) -> None:
        self._image_bgr: np.ndarray | None = None
        self._working_bgr: np.ndarray | None = None
        self._last_scale_rect: tuple[int, int, int, int] | None = None

    def load_image(self, path: str | Path) -> np.ndarray:
        """Load BMP via OpenCV; ensure BGR 3-channel."""
        path = Path(path)
        if path.suffix.lower() != ".bmp":
            raise ValueError(f"Expected .bmp path, got: {path.suffix!r}")
        img = imread_bgr(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self._image_bgr = img
        self._working_bgr = img.copy()
        return img

    def _ensure_image(self) -> np.ndarray:
        if self._working_bgr is None:
            raise RuntimeError("No image loaded. Call load_image first.")
        return self._working_bgr

    def build_color_map(
        self,
        scale_colors: np.ndarray,
        min_temp: float,
        max_temp: float,
    ) -> dict[str, Any]:
        """Structure passed to get_pixel_temp."""
        if scale_colors.size == 0:
            raise ValueError("Empty scale_colors")
        sc = scale_colors.astype(np.float32)
        lab = cv2.cvtColor(
            sc.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
        ).astype(np.float32).reshape(-1, 3)
        return {
            "scale_colors": sc,
            "scale_lab": lab,
            "min_temp": float(min_temp),
            "max_temp": float(max_temp),
            "n": int(len(sc)),
        }

    def get_pixel_temp(
        self,
        x: int,
        y: int,
        color_map: dict[str, Any],
    ) -> float | None:
        """Temperature at pixel or None if INVALID."""
        img = self._ensure_image()
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            return None
        bgr = img[y, x].astype(np.float32)
        return self._single_bgr_temp(bgr, color_map)

    def _single_bgr_temp(
        self,
        bgr: np.ndarray,
        color_map: dict[str, Any],
    ) -> float | None:
        sc: np.ndarray = color_map["scale_colors"]
        lab_sc: np.ndarray = color_map["scale_lab"]
        n = color_map["n"]
        min_t = color_map["min_temp"]
        max_t = color_map["max_temp"]

        dist = np.sqrt(np.sum((sc - bgr.reshape(1, 3)) ** 2, axis=1))
        idx = int(np.argmin(dist))
        min_d = float(dist[idx])

        if min_d > C.bgr_near_scale_thresh:
            return None
        if min_d >= C.bgr_ambiguous_low:
            px_lab = cv2.cvtColor(
                bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB
            ).astype(np.float32).reshape(3)
            de = float(np.linalg.norm(px_lab - lab_sc[idx]))
            if de > C.gradient_delta_e_floor_soft:
                return None

        if self._is_text_pixel_bgr(bgr):
            if min_d > C.text_mask_min_bgr_dist:
                return None

        if n <= 1:
            t = 0.5 * (min_t + max_t)
            return float(t)
        # Top of scale row 0 = max_temp, bottom = min_temp
        frac = idx / float(n - 1)
        temp = max_t + frac * (min_t - max_t)
        return float(temp)

    def _is_text_pixel_bgr(self, bgr: np.ndarray) -> bool:
        px = bgr.reshape(1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(3)
        H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(3)
        L = lab[0] * 100.0 / 255.0
        a = float(lab[1]) - 128.0
        b = float(lab[2]) - 128.0
        chroma = abs(a) + abs(b)

        hsv_extreme = (V >= C.hsv_white_v_min and S <= C.hsv_chroma_max_for_extreme) or (
            V <= C.hsv_black_v_max and S <= C.hsv_chroma_max_for_extreme
        )
        lab_light = L >= C.lab_l_white_min and chroma <= C.lab_ab_chroma_max_neutral * 2
        lab_dark = L <= C.lab_l_black_max and chroma <= C.lab_ab_chroma_max_neutral * 2
        return bool(hsv_extreme or lab_light or lab_dark)

    def _text_mask_image(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2].astype(np.int16)
        S = hsv[:, :, 1].astype(np.int16)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0] * 100.0 / 255.0
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        chroma = np.abs(a) + np.abs(b)
        hsv_extreme = ((V >= C.hsv_white_v_min) & (S <= C.hsv_chroma_max_for_extreme)) | (
            (V <= C.hsv_black_v_max) & (S <= C.hsv_chroma_max_for_extreme)
        )
        lab_light = (L >= C.lab_l_white_min) & (
            chroma <= C.lab_ab_chroma_max_neutral * 2
        )
        lab_dark = (L <= C.lab_l_black_max) & (
            chroma <= C.lab_ab_chroma_max_neutral * 2
        )
        return hsv_extreme | lab_light | lab_dark

    def calculate_temperature_matrix(
        self,
        scale_colors: np.ndarray,
        min_temp: float,
        max_temp: float,
        *,
        scale_rect: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Full temperature matrix; scale ROI and INVALID pixels are NaN."""
        img = self._ensure_image()
        color_map = self.build_color_map(scale_colors, min_temp, max_temp)
        h, w = img.shape[:2]
        flat = img.reshape(-1, 3).astype(np.float32)
        sc = color_map["scale_colors"]
        lab_sc = color_map["scale_lab"]
        n = color_map["n"]
        text_mask = self._text_mask_image(img).reshape(-1)

        out = np.full((flat.shape[0],), np.nan, dtype=np.float64)
        chunk = 4096
        for start in range(0, flat.shape[0], chunk):
            end = min(start + chunk, flat.shape[0])
            px = flat[start:end]
            d = np.sqrt(
                np.sum((px[:, None, :] - sc[None, :, :]) ** 2, axis=2)
            )
            idx = np.argmin(d, axis=1)
            min_d = d[np.arange(end - start), idx]

            invalid = min_d > C.bgr_near_scale_thresh
            amb = (~invalid) & (min_d >= C.bgr_ambiguous_low)
            if np.any(amb):
                sub = px[amb]
                labs = (
                    cv2.cvtColor(sub.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
                    .astype(np.float32)
                    .reshape(-1, 3)
                )
                pick = lab_sc[idx[amb]]
                de = np.linalg.norm(labs - pick, axis=1)
                invalid_amb = de > C.gradient_delta_e_floor_soft
                full_idx = np.where(amb)[0]
                invalid[full_idx[invalid_amb]] = True

            tm = text_mask[start:end]
            near = (~invalid) & tm & (min_d > C.text_mask_min_bgr_dist)
            invalid |= near

            valid = ~invalid
            if n <= 1:
                temp_val = 0.5 * (min_temp + max_temp)
                out[start:end][valid] = temp_val
            else:
                frac = idx[valid].astype(np.float64) / float(n - 1)
                temps = max_temp + frac * (min_temp - max_temp)
                out[start:end][valid] = temps

        mat = out.reshape(h, w)
        if scale_rect is not None:
            sx, sy, sw, sh = scale_rect
            sx = max(0, sx)
            sy = max(0, sy)
            ex = min(w, sx + sw)
            ey = min(h, sy + sh)
            mat[sy:ey, sx:ex] = np.nan
        return mat

    def scan_ocv(
        self,
        image_path: str | Path,
        *,
        output_dir: str | Path,
        scale_rect: tuple[int, int, int, int] | None,
        min_temp: float,
        max_temp: float,
        apply_blur: bool = False,
        auto_detect_scale: bool = True,
        gradient_strict: bool = True,
        overlay_mode: str = "both",
        colormap_name: str = "JET",
        grid_step: int | None = None,
        include_temp_matrix: bool = True,
    ) -> dict[str, Any]:
        """
        Full pipeline: load -> scale ROI -> validate -> matrix -> export.
        If auto_detect_scale and scale_rect is None, runs ScaleDetector.
        """
        image_path = Path(image_path)
        self.load_image(image_path)
        assert self._image_bgr is not None
        base = self._image_bgr
        self._working_bgr = base.copy()

        if apply_blur:
            k = C.gaussian_ksize
            self._working_bgr = cv2.GaussianBlur(
                self._working_bgr,
                (k, k),
                C.gaussian_sigma,
            )

        det = ScaleDetector()
        if scale_rect is not None:
            rect = scale_rect
        elif auto_detect_scale:
            rect = det.detect_scale(self._working_bgr)
        else:
            rect = None
        if rect is None:
            raise RuntimeError("Шкала не найдена: задайте область вручную.")

        x, y, rw, rh = rect
        h, w = self._working_bgr.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        rect = (x, y, rw, rh)
        self._last_scale_rect = rect

        roi = self._working_bgr[y : y + rh, x : x + rw]
        colors = det.get_scale_colors(roi)
        det.validate_scale_gradient(colors, strict=gradient_strict)

        temp_matrix = self.calculate_temperature_matrix(
            colors, min_temp, max_temp, scale_rect=rect
        )
        invalid_count = int(np.sum(~np.isfinite(temp_matrix)))

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = out_dir / "result_overlay.bmp"
        csv_path = out_dir / "data.csv"

        step = grid_step if grid_step is not None else C.default_grid_step_px
        overlay_gray = build_result_overlay_file(
            base,
            temp_matrix,
            grid_step=step,
            scale_rect=rect,
            max_invalid_markers=C.max_invalid_overlay_markers,
        )
        if not imwrite_image(overlay_path, overlay_gray):
            raise OSError(f"Cannot write: {overlay_path}")
        save_temperature_csv(csv_path, temp_matrix)

        result: dict[str, Any] = {
            "image_path": str(image_path),
            "output_dir": str(out_dir),
            "result_overlay": str(overlay_path),
            "data_csv": str(csv_path),
            "scale_rect": rect,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "invalid_pixel_count": invalid_count,
        }
        if include_temp_matrix:
            result["temp_matrix"] = temp_matrix
        return result

"""Save CSV and overlay BMP."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from thermal_decoder import constants as C


def save_temperature_csv(
    path: str | Path,
    temp_matrix: np.ndarray,
    *,
    chunk_rows: int | None = None,
) -> None:
    """Write X,Y,Temperature UTF-8 CSV; INVALID as empty temperature cell."""
    path = Path(path)
    chunk_rows = chunk_rows or C.csv_chunk_rows
    h, w = temp_matrix.shape
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Temperature"])
        buf = []
        for y in range(h):
            row = temp_matrix[y]
            for x in range(w):
                v = row[x]
                if np.isnan(v):
                    buf.append([x, y, ""])
                else:
                    buf.append([x, y, f"{float(v):.6f}"])
                if len(buf) >= chunk_rows:
                    writer.writerows(buf)
                    buf.clear()
        if buf:
            writer.writerows(buf)


def _colormap_id(name: str) -> int:
    n = name.upper()
    cmap = getattr(cv2, f"COLORMAP_{n}", None)
    if cmap is None:
        return cv2.COLORMAP_JET
    return int(cmap)


def _bw_fg_bg(gray: np.ndarray, x: int, y: int) -> tuple[int, int]:
    """(foreground, background) scalars for drawing on grayscale: high contrast."""
    h, w = gray.shape[:2]
    if 0 <= y < h and 0 <= x < w:
        v = int(gray[y, x])
    else:
        v = 128
    if v >= 128:
        return 0, 255
    return 255, 0


def build_result_overlay_file(
    base_bgr: np.ndarray,
    temp_matrix: np.ndarray,
    *,
    grid_step: int,
    scale_rect: tuple[int, int, int, int] | None,
    draw_invalid_markers: bool = True,
    max_invalid_markers: int = 5000,
) -> np.ndarray:
    """
    Single-channel grayscale BMP export: no automatic labels or INVALID markers.
    Use the «Просмотр температур» window to place markers manually, then save.
    """
    _ = (
        temp_matrix,
        grid_step,
        scale_rect,
        draw_invalid_markers,
        max_invalid_markers,
    )
    return cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)


def build_overlay(
    base_bgr: np.ndarray,
    temp_matrix: np.ndarray,
    *,
    overlay_mode: str,
    colormap_name: str,
    grid_step: int,
    scale_rect: tuple[int, int, int, int] | None,
    draw_invalid_markers: bool = True,
    max_invalid_markers: int = 5000,
) -> np.ndarray:
    """
    overlay_mode: 'grid' | 'colormap' | 'both'
    Top of scale = max temp (matrix already mapped that way).
    """
    out = base_bgr.copy()
    h, w = out.shape[:2]
    valid = np.isfinite(temp_matrix)
    invalid = ~valid

    if overlay_mode in ("colormap", "both"):
        t = temp_matrix.copy()
        if not np.any(np.isfinite(t)):
            pass
        else:
            tmin = np.nanmin(t)
            tmax = np.nanmax(t)
            if tmax > tmin:
                norm = (t - tmin) / (tmax - tmin)
            else:
                norm = np.zeros_like(t, dtype=np.float64)
            norm_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
            cmap_id = _colormap_id(colormap_name)
            color = cv2.applyColorMap(norm_u8, cmap_id)
            mask = np.isfinite(temp_matrix)
            blend = out.astype(np.float32)
            col = color.astype(np.float32)
            alpha = 0.45
            for c in range(3):
                blend[:, :, c] = np.where(
                    mask,
                    (1 - alpha) * blend[:, :, c] + alpha * col[:, :, c],
                    blend[:, :, c],
                )
            out = np.clip(blend, 0, 255).astype(np.uint8)

    if overlay_mode in ("grid", "both"):
        step = max(8, int(grid_step))
        font = cv2.FONT_HERSHEY_SIMPLEX
        for y in range(0, h, step):
            for x in range(0, w, step):
                if not np.isfinite(temp_matrix[y, x]):
                    continue
                val = float(temp_matrix[y, x])
                label = f"{val:.1f}"
                cv2.putText(
                    out,
                    label,
                    (x + 2, y + 14),
                    font,
                    0.35,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    label,
                    (x + 2, y + 14),
                    font,
                    0.35,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    # INVALID markers: cross + ERR (прореживаем при массовом INVALID)
    if draw_invalid_markers:
        ys, xs = np.where(invalid)
        nmark = int(xs.size)
        if (
            max_invalid_markers > 0
            and nmark > max_invalid_markers
        ):
            stride = max(1, int(np.ceil(nmark / max_invalid_markers)))
            sel = np.arange(0, nmark, stride, dtype=np.intp)
            xs = xs[sel]
            ys = ys[sel]
        for y, x in zip(ys.tolist(), xs.tolist()):
            if scale_rect is not None:
                sx, sy, sw, sh = scale_rect
                if sx <= x < sx + sw and sy <= y < sy + sh:
                    continue
            c = C.invalid_cross_color_bgr
            t = C.invalid_cross_thickness
            s = 5
            cv2.line(out, (x - s, y - s), (x + s, y + s), c, t, cv2.LINE_AA)
            cv2.line(out, (x - s, y + s), (x + s, y - s), c, t, cv2.LINE_AA)
            cv2.putText(
                out,
                C.invalid_err_text,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                c,
                1,
                cv2.LINE_AA,
            )

    return out

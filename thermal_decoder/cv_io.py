"""OpenCV image I/O with Unicode paths on Windows.

`cv2.imread` / `cv2.imwrite` use legacy APIs and fail for paths with non-ASCII
characters (e.g. Cyrillic in user or folder names). Read/write bytes via
`Path.open` (UTF-8 paths on Windows) and use `imdecode` / `imencode`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread_bgr", "imwrite"]


def _numpy_to_bgr_u8(img: np.ndarray) -> np.ndarray | None:
    """Turn OpenCV imdecode / Pillow array into BGR uint8, or return None if unsupported."""
    if img is None or img.size == 0:
        return None
    c = 1
    if img.ndim == 3:
        c = int(img.shape[2])
    # --- uint8: common case
    if img.dtype == np.uint8:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if c == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if c == 3:
            return img
        return None
    # --- 16-bit BMP / 16 bpc: OpenCV may return uint16
    if img.dtype == np.uint16:
        u8 = (img >> 8).astype(np.uint8)
        if u8.ndim == 2:
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        if c == 4:
            return cv2.cvtColor(u8, cv2.COLOR_BGRA2BGR)
        if c == 3:
            return u8
        return None
    # --- float / double (rare in BMP, but imdecode can yield float)
    if img.dtype in (np.float32, np.float64):
        if img.ndim == 2:
            u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if u8 is None:
                return None
            return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        if c == 3:
            u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return u8
        if c == 4:
            u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.cvtColor(u8, cv2.COLOR_BGRA2BGR)
        return None
    return None


def _imread_bgr_pillow(path: Path) -> np.ndarray | None:
    """Second attempt: Pillow reads some BMPs that OpenCV cannot (e.g. certain RLE / formats)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
        a = np.asarray(rgb, dtype=np.uint8)
        if a.ndim != 3 or a.shape[2] != 3:
            return None
        return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, TypeError, NotImplementedError, RuntimeError, MemoryError):
        return None


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """Read image in BGR; return None on failure. Same channel handling as previous imread+load_image."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            raw = f.read()
    except OSError:
        return None
    if not raw:
        return None
    data = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    out = _numpy_to_bgr_u8(img) if img is not None else None
    if out is not None:
        return out
    return _imread_bgr_pillow(p)


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Write image; extension (e.g. .bmp) must match *path*."""
    p = Path(path)
    ext = (p.suffix if p.suffix else ".bmp").lower()
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    ok, buf = cv2.imencode(ext, img)
    if not ok or buf is None:
        return False
    try:
        with p.open("wb") as f:
            f.write(buf.tobytes())
    except OSError:
        return False
    return True

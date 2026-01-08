# circle_bundles/viz/image_utils.py
from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

__all__ = ["fig_to_rgba", "trim_image", "render_to_rgba"]


def fig_to_rgba(fig: Figure) -> np.ndarray:
    """Render a Matplotlib figure to an (H,W,4) uint8 RGBA array."""
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf


def trim_image(img: np.ndarray, *, white_thresh: int = 250) -> np.ndarray:
    """
    Trim whitespace around an image array.

    - If RGBA: trims where alpha==0
    - If RGB: trims where all channels >= white_thresh
    - If grayscale: trims where value >= white_thresh
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        mask = arr < int(white_thresh)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[..., 3] > 0
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mask = np.any(arr < int(white_thresh), axis=2)
    else:
        raise ValueError(
            f"Unsupported image shape {arr.shape}. Expected (H,W), (H,W,3), or (H,W,4)."
        )

    if not np.any(mask):
        return arr

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return arr[y0:y1, x0:x1]


def _ensure_uint8_rgba(img: np.ndarray) -> np.ndarray:
    """Coerce image to uint8 RGBA."""
    arr = np.asarray(img)

    # floats -> uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {arr.shape} after coercion.")

    if arr.shape[2] == 3:
        alpha = 255 * np.ones((*arr.shape[:2], 1), dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)

    return arr


def _transparent_outer_border_rgba(
    rgba_u8: np.ndarray,
    *,
    bg_rgb: Optional[Tuple[int, int, int]] = None,
    tol: int = 2,
    alpha_bg_max: int = 5,   # NEW: only treat near-transparent as background
) -> np.ndarray:
    """
    Make only the *outer* uniform background transparent (RGBA uint8),
    but do NOT erase drawn content (like borders).

    We only consider a row/col to be removable background if:
      1) its RGB is close to bg (within tol), AND
      2) its alpha is already near-transparent (<= alpha_bg_max).
    """
    img = np.array(rgba_u8, copy=True)
    rgb = img[..., :3]
    a = img[..., 3]

    if bg_rgb is None:
        bg = rgb[0, 0].astype(int)
    else:
        bg = np.array(bg_rgb, dtype=int)

    def row_is_bg(r: int) -> bool:
        rgb_ok = np.all(np.abs(rgb[r].astype(int) - bg) <= tol)
        a_ok = np.max(a[r, :]) <= alpha_bg_max
        return bool(rgb_ok and a_ok)

    def col_is_bg(c: int) -> bool:
        rgb_ok = np.all(np.abs(rgb[:, c].astype(int) - bg) <= tol)
        a_ok = np.max(a[:, c]) <= alpha_bg_max
        return bool(rgb_ok and a_ok)

    h, w = rgb.shape[:2]

    top = 0
    while top < h and row_is_bg(top):
        a[top, :] = 0
        top += 1

    bottom = h - 1
    while bottom >= 0 and row_is_bg(bottom):
        a[bottom, :] = 0
        bottom -= 1

    left = 0
    while left < w and col_is_bg(left):
        a[:, left] = 0
        left += 1

    right = w - 1
    while right >= 0 and col_is_bg(right):
        a[:, right] = 0
        right -= 1

    img[..., 3] = a
    return img



def render_to_rgba(
    obj: Union[np.ndarray, Figure],
    *,
    transparent_border: bool = False,
    bg_rgb: Optional[Tuple[int, int, int]] = None,
    border_tol: int = 2,
    trim: bool = False,
    white_thresh: int = 250,
) -> np.ndarray:
    """
    Canonical renderer: returns (H,W,4) uint8 RGBA no matter what.

    - If obj is a Figure: render via Agg.
    - If obj is ndarray: coerce to uint8 RGBA.
    - If transparent_border: punch out only the outer uniform background.
    - If trim: crop based on alpha (RGBA) or white_thresh (RGB/gray).
    """
    if isinstance(obj, Figure):
        rgba = fig_to_rgba(obj)
        plt.close(obj)
    else:
        rgba = _ensure_uint8_rgba(obj)

    if transparent_border:
        rgba = _transparent_outer_border_rgba(rgba, bg_rgb=bg_rgb, tol=int(border_tol))

    if trim:
        rgba = trim_image(rgba, white_thresh=int(white_thresh))

    return rgba

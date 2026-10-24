"""Corruption suite for uTHCD-C.

Ten corruption types, each at five severity levels (1 = mildest, 5 = strongest):
nine general image corruptions adapted from established corruption benchmarks and
one handwriting-specific stroke-thinning operator. Corruptions operate on a PIL
image at its native resolution and return a corrupted PIL image; resizing and
normalization for the network happen afterwards (see ``uthcd_c.data``).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
from skimage.morphology import skeletonize

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "gaussian_blur", "defocus_blur",
    "stroke_thinning",
    "elastic", "pixelate", "contrast", "scale",
]
SEVERITIES = [1, 2, 3, 4, 5]


def _clamp(arr: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(arr, 0, 255))


def _gaussian_noise(img, s):
    arr = np.asarray(img, dtype=np.float32)
    sigma = [10, 20, 30, 40, 60][s - 1]
    return Image.fromarray(_clamp(arr + np.random.normal(0, sigma, arr.shape)))


def _shot_noise(img, s):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    scale = [60, 40, 25, 15, 8][s - 1]
    noisy = np.random.poisson(arr * scale) / float(scale)
    return Image.fromarray(_clamp(noisy * 255.0))


def _impulse_noise(img, s):
    arr = np.asarray(img).copy()
    p = [0.03, 0.06, 0.12, 0.18, 0.25][s - 1]
    mask = np.random.choice([0, 1, 2], size=arr.shape[:2], p=[1 - p, p / 2, p / 2])
    arr[mask == 1] = 0
    arr[mask == 2] = 255
    return Image.fromarray(arr)


def _gaussian_blur(img, s):
    radius = [0.7, 1.5, 2.5, 3.5, 5.0][s - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _defocus_blur(img, s):
    radius = [1, 2, 3, 4, 6][s - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _stroke_thinning(img, s):
    """Binarize, reduce strokes to a 1-pixel skeleton (Zhang--Suen), then
    re-dilate with a kernel whose size controls the resulting stroke width.
    A smaller kernel leaves thinner strokes (stronger corruption)."""
    max_kernel = 9
    k = int(round(max_kernel * (1.0 - (s - 1) / 4.0)))
    if k % 2 == 0:
        k = max(1, k - 1)
    dilate_size = max(1, k)
    rgb = img.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    thresh = gray.mean() - k * gray.std()
    mask = (gray < thresh).astype(np.uint8)
    if mask.sum() == 0:
        mask = (gray < gray.mean()).astype(np.uint8)
    skeleton = skeletonize(mask.astype(bool), method="zhang").astype(np.uint8)
    sk_pil = Image.fromarray((skeleton * 255).astype(np.uint8))
    dilated = np.asarray(sk_pil.filter(ImageFilter.MaxFilter(size=dilate_size)), dtype=np.uint8)
    fg = dilated == 255
    out = np.asarray(rgb, dtype=np.uint8).copy()
    out[~np.stack([fg, fg, fg], axis=-1)] = 255
    return Image.fromarray(out)


def _contrast(img, s):
    factor = [0.7, 0.5, 0.35, 0.2, 0.1][s - 1]
    arr = np.asarray(img, dtype=np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    return Image.fromarray(_clamp((arr - mean) * factor + mean))


def _pixelate(img, s):
    frac = [0.9, 0.7, 0.5, 0.33, 0.2][s - 1]
    w, h = img.size
    small = img.resize((max(1, int(w * frac)), max(1, int(h * frac))), Image.NEAREST)
    return small.resize(img.size, Image.NEAREST)


def _elastic(img, s):
    shift = [1, 2, 3, 4, 6][s - 1]
    dx = np.random.uniform(-shift, shift)
    dy = np.random.uniform(-shift, shift)
    angle = np.random.uniform(-shift * 2, shift * 2)
    warped = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))
    return warped.rotate(angle, resample=Image.BILINEAR)


def _scale(img, s):
    target = {1: 128, 2: 64, 3: 32, 4: 16, 5: 8}[s]
    w, h = img.size
    small = img.resize((target, target), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


_FUNCS = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "gaussian_blur": _gaussian_blur,
    "defocus_blur": _defocus_blur,
    "stroke_thinning": _stroke_thinning,
    "elastic": _elastic,
    "pixelate": _pixelate,
    "contrast": _contrast,
    "scale": _scale,
}


def apply_corruption(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    """Return ``img`` with ``corruption`` applied at the given ``severity`` (1-5).

    ``corruption='clean'`` (or ``None``) returns the image unchanged.
    """
    if corruption in ("clean", None):
        return img
    if corruption not in _FUNCS:
        raise ValueError(f"Unknown corruption '{corruption}'. Choose from {CORRUPTIONS}.")
    severity = max(1, min(5, int(severity)))
    return _FUNCS[corruption](img, severity)

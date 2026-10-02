"""Shared validation for public numerical entry points."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def matrix(value: ArrayLike, name: str, shape: tuple[int, int] | None = None) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    result = np.asarray(raw, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty, finite real matrix")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}; expected {shape}")
    return result


def positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def scalar(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def time_grid(value: ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if (result.ndim != 1 or len(result) < 2 or not np.all(np.isfinite(result))
            or result[0] != 0 or not np.all(np.diff(result) > 0)):
        raise ValueError("times must start at zero and be finite and strictly increasing")
    return result


def psd_eigh(value: ArrayLike, *, rtol: float = 1e-12) -> tuple[FloatArray, FloatArray]:
    """Check symmetry/PSD relatively; only roundoff-level negative values are clipped."""
    a = matrix(value, "covariance")
    scalar(rtol, "rtol", allow_zero=True)
    if a.shape[0] != a.shape[1]:
        raise ValueError("covariance must be square")
    scale = float(np.max(np.abs(a)))
    if np.max(np.abs(a - a.T)) > rtol * scale:
        raise ValueError("covariance must be symmetric")
    values, vectors = np.linalg.eigh((a + a.T) / 2)
    if values[0] < -rtol * max(float(np.max(np.abs(values))), scale):
        raise ValueError("covariance must be positive semidefinite")
    return np.maximum(values, 0), vectors

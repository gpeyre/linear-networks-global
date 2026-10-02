"""Predictor losses. All predictors have shape (d_v, d_u)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from ._validation import FloatArray, matrix, scalar


class PredictorLoss(Protocol):
    """Minimal custom-loss interface; smoothness/PL are not inferred by a solver."""
    shape: tuple[int, int]

    def value(self, predictor: ArrayLike) -> float: ...
    def gradient(self, predictor: ArrayLike) -> FloatArray: ...


def _readonly(value: ArrayLike, name: str) -> FloatArray:
    result = matrix(value, name).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class QuadraticLoss:
    """ell(S) = sum(weights * (S-target)^2) / 2.

    Zero weights model an observation kernel. The PL constant is the smallest
    strictly positive weight, or zero for the constant objective.
    """
    target: ArrayLike
    weights: ArrayLike | None = None

    def __post_init__(self) -> None:
        target = _readonly(self.target, "target")
        weights = np.ones_like(target) if self.weights is None else matrix(self.weights, "weights", target.shape)
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "weights", _readonly(weights, "weights"))

    @property
    def shape(self) -> tuple[int, int]:
        return self.target.shape

    @property
    def kappa(self) -> float:
        positive = self.weights[self.weights > 0]
        return float(positive.min()) if positive.size else 0.0

    @property
    def smoothness(self) -> float:
        return float(self.weights.max())

    @property
    def minimum(self) -> float:
        return 0.0

    def value(self, predictor: ArrayLike) -> float:
        residual = matrix(predictor, "predictor", self.shape) - self.target
        return float(0.5 * np.sum(self.weights * residual**2))

    def gradient(self, predictor: ArrayLike) -> FloatArray:
        return self.weights * (matrix(predictor, "predictor", self.shape) - self.target)

    def gap(self, predictor: ArrayLike) -> float:
        return self.value(predictor)


def smooth_lasso_gap(predictor: ArrayLike, optimum: ArrayLike, strength: float, smoothing: float) -> FloatArray:
    """Cancellation-resistant gaps, optionally batched over leading axes.

    ``optimum`` must be a stationary minimizer of the same smoothed-Lasso loss.
    For opposite signs the rationalized denominator is evaluated separately
    to avoid catastrophic cancellation when smoothing is small.
    """
    scalar(strength, "strength", allow_zero=True)
    scalar(smoothing, "smoothing")
    x, z = np.broadcast_arrays(np.asarray(predictor, dtype=float), np.asarray(optimum, dtype=float))
    if x.ndim < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
        raise ValueError("predictor and optimum must be finite matrices or batches")
    difference = x - z
    hx, hz = np.hypot(x, smoothing), np.hypot(z, smoothing)
    same = x * z >= 0
    bregman = np.empty_like(x)
    bregman[same] = (smoothing**2 * difference[same]**2
                     / (hz[same] * (hx[same] * hz[same] + x[same] * z[same] + smoothing**2)))
    opposite = ~same
    denominator = hz[opposite] * (1 + (x[opposite]**2 + z[opposite]**2 + smoothing**2)
                                  / (hx[opposite] * hz[opposite] - x[opposite] * z[opposite]))
    bregman[opposite] = difference[opposite]**2 / denominator
    return np.sum(0.5 * difference**2 + strength * bregman, axis=(-2, -1))


@dataclass(frozen=True)
class SmoothedLassoLoss:
    """Quadratic fitting plus strength * sum(hypot(S, smoothing)-smoothing)."""
    target: ArrayLike
    strength: float = 0.1
    smoothing: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _readonly(self.target, "target"))
        object.__setattr__(self, "strength", scalar(self.strength, "strength", allow_zero=True))
        object.__setattr__(self, "smoothing", scalar(self.smoothing, "smoothing"))

    @property
    def shape(self) -> tuple[int, int]:
        return self.target.shape

    @property
    def kappa(self) -> float:
        return 1.0

    @property
    def smoothness(self) -> float:
        return 1 + self.strength / self.smoothing

    def value(self, predictor: ArrayLike) -> float:
        x = matrix(predictor, "predictor", self.shape)
        return float(np.sum(0.5 * (x - self.target)**2
                            + self.strength * (np.hypot(x, self.smoothing) - self.smoothing)))

    def gradient(self, predictor: ArrayLike) -> FloatArray:
        x = matrix(predictor, "predictor", self.shape)
        return x - self.target + self.strength * x / np.hypot(x, self.smoothing)

    @cached_property
    def minimizer(self) -> FloatArray:
        result = np.empty_like(self.target)
        for index in np.ndindex(self.shape):
            target = self.target[index]
            result[index] = 0 if target == 0 else brentq(
                lambda x: x - target + self.strength * x / np.hypot(x, self.smoothing),
                min(0, target), max(0, target), xtol=np.nextafter(0.0, 1.0),
                rtol=4 * np.finfo(float).eps)
        result.flags.writeable = False
        return result

    @cached_property
    def minimum(self) -> float:
        return self.value(self.minimizer)

    def gap(self, predictor: ArrayLike) -> float:
        x = matrix(predictor, "predictor", self.shape)
        return float(smooth_lasso_gap(x, self.minimizer, self.strength, self.smoothing))

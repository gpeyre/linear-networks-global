"""Initialization-only momentum certificates, not fitted dynamical rates."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, scalar


def _loss_constants(kappa: float, smoothness: float, energy: float) -> None:
    scalar(kappa, "kappa")
    scalar(smoothness, "smoothness")
    scalar(energy, "energy", allow_zero=True)
    if kappa > smoothness:
        raise ValueError("kappa cannot exceed smoothness for a nonconstant smooth PL loss")


def white_bootstrap(s: ArrayLike, sigma_vu: float, damping: ArrayLike, *,
                    kappa: float, smoothness: float, energy: float) -> tuple[FloatArray, FloatArray]:
    """Return (deformation gap, conservative energy-decay rate).

    Applies to initial position covariance sigma_vu^2 I, zero position-velocity
    cross covariance, and positive damping. ``energy`` is the initial loss gap
    plus half the velocity trace. A strictly negative gap certifies a tube;
    a nonnegative gap is inconclusive, not evidence of divergence. Inputs s and
    damping broadcast. The spectral budget s must be strictly positive.
    """
    _loss_constants(kappa, smoothness, energy)
    sigma = scalar(sigma_vu, "sigma_vu")
    s, gamma = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(damping, dtype=float))
    if not np.all(np.isfinite(s)) or not np.all(s > 0) or not np.all(np.isfinite(gamma)) or not np.all(gamma > 0):
        raise ValueError("s and damping must be finite and positive")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        a, b = sigma**2 * np.exp(-2*s), sigma**2 * np.exp(2*s)
        rho = 4*b*smoothness + 2*np.sqrt(2*smoothness*energy)
        rate = a*kappa*gamma/(rho + 2*gamma**2)
        gap = 4/rate*np.sqrt(3*energy/(2*a)) - s
    return gap, rate


def tightened_momentum_rate(a: float, b: float, damping: float, *,
                             kappa: float, smoothness: float, energy: float) -> tuple[float, float]:
    """Return (rate, q) on an already certified position tube a I <= Sigma <= b I.

    The energy envelope is energy/(1-q) * exp(-rate*t). This formula alone
    does not certify the tube; establish it first, e.g. with white_bootstrap.
    """
    _loss_constants(kappa, smoothness, energy)
    a, b, gamma = scalar(a, "a"), scalar(b, "b"), scalar(damping, "damping")
    if a > b:
        raise ValueError("tube bounds must satisfy a <= b")
    force_pl, force_smooth = 4*a*kappa, 4*b*smoothness
    rho = force_smooth + 2*np.sqrt(2*smoothness*energy)
    eta = gamma/(gamma + np.sqrt(gamma**2 + 2*rho))
    aa, bb = (1-eta)*force_pl, rho + gamma**2/(2*eta)
    epsilon = 2*gamma/(aa+bb)
    q = epsilon*np.sqrt(force_smooth/2)
    rate = epsilon*aa/(1+q)
    if not np.isfinite(rate) or rate <= 0 or not 0 < q < 1:
        raise ValueError("parameters do not give a finite positive rate with 0 < q < 1")
    return float(rate), float(q)

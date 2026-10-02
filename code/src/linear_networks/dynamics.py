"""Mean-field-time factor, covariance, and heavy-ball vector fields.

    W' = -G W,  Sigma' = -G Sigma - Sigma G,
    W'' + gamma W' = -G W.

These are the factor flows for n * ell(V U.T / n), not ell alone.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, matrix, scalar
from .losses import PredictorLoss


def force_matrix(gradient: ArrayLike) -> FloatArray:
    """G(E) with off-diagonal blocks E.T and E, for arbitrary dimensions."""
    e = matrix(gradient, "gradient")
    d_v, d_u = e.shape
    return np.block([[np.zeros((d_u, d_u)), e.T], [e, np.zeros((d_v, d_v))]])


def factor_rhs(factors: ArrayLike, loss: PredictorLoss) -> FloatArray:
    """Velocity of W=(U;V); the updates use the same, unmodified state."""
    d_v, d_u = loss.shape
    w = matrix(factors, "factors")
    if w.shape[0] != d_u + d_v:
        raise ValueError("factor rows must equal d_u + d_v")
    u, v = w[:d_u], w[d_u:]
    e = matrix(loss.gradient(v @ u.T / w.shape[1]), "loss gradient", loss.shape)
    return np.vstack((-e.T @ v, -e @ u))


def covariance_rhs(covariance: ArrayLike, loss: PredictorLoss) -> FloatArray:
    """Bures covariance velocity; the RHS does not project onto the PSD cone."""
    d_v, d_u = loss.shape
    m = d_u + d_v
    a = matrix(covariance, "covariance", (m, m))
    a = (a + a.T) / 2
    g = force_matrix(matrix(loss.gradient(a[d_u:, :d_u]), "loss gradient", loss.shape))
    result = -(g @ a + a @ g)
    return (result + result.T) / 2


def factor_step(factors: ArrayLike, loss: PredictorLoss, step_size: float) -> FloatArray:
    """One simultaneous factor-GD step with mean-field-time stepsize tau."""
    tau = scalar(step_size, "step_size")
    w = matrix(factors, "factors")
    return w + tau * factor_rhs(w, loss)


def covariance_step(covariance: ArrayLike, loss: PredictorLoss, step_size: float) -> FloatArray:
    """Exact covariance of factor GD: (I-tau G) Sigma (I-tau G)."""
    tau = scalar(step_size, "step_size")
    d_v, d_u = loss.shape
    m = d_u + d_v
    a = matrix(covariance, "covariance", (m, m))
    g = force_matrix(matrix(loss.gradient(a[d_u:, :d_u]), "loss gradient", loss.shape))
    transition = np.eye(m) - tau * g
    return transition @ a @ transition.T


def phase_covariance_rhs(phase: ArrayLike, loss: PredictorLoss, damping: float) -> FloatArray:
    """Covariance velocity in (u,v,p,q) order, with (p,q)=(u',v')."""
    gamma = scalar(damping, "damping", allow_zero=True)
    d_v, d_u = loss.shape
    m = d_u + d_v
    a = matrix(phase, "phase covariance", (2*m, 2*m))
    a = (a + a.T) / 2
    g = force_matrix(matrix(loss.gradient(a[d_u:m, :d_u]), "loss gradient", loss.shape))
    drift = np.block([[np.zeros((m, m)), np.eye(m)], [-g, -gamma * np.eye(m)]])
    return drift @ a + a @ drift.T

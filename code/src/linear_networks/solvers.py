"""Adaptive ODE solvers and true discrete factor gradient descent."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from ._validation import FloatArray, matrix, psd_eigh, scalar, time_grid
from .dynamics import covariance_rhs, factor_rhs, factor_step, phase_covariance_rhs
from .losses import PredictorLoss


@dataclass
class Trajectory:
    """Arrays sampled at requested times; no PSD or loss-tail clipping is applied.

    ``loss_gaps`` is None for a custom loss without a ``gap`` method. Factor
    and phase paths are present only for their corresponding integrator.
    """
    times: FloatArray
    covariances: FloatArray
    predictors: FloatArray
    objectives: FloatArray
    loss_gaps: FloatArray | None
    factors: FloatArray | None = None
    phase_covariances: FloatArray | None = None

    @property
    def energies(self) -> FloatArray:
        """Mechanical excess energy, requiring a known gap and a phase path."""
        if self.phase_covariances is None or self.loss_gaps is None:
            raise ValueError("mechanical excess energy needs phase covariances and loss gaps")
        m = self.covariances.shape[-1]
        return self.loss_gaps + 0.5 * np.trace(self.phase_covariances[:, m:, m:], axis1=1, axis2=2)


def _result(times, covariances, loss, *, factors=None, phase=None) -> Trajectory:
    d_u = loss.shape[1]
    predictors = covariances[:, d_u:, :d_u].copy()
    objectives = np.array([loss.value(s) for s in predictors])
    gap = getattr(loss, "gap", None)
    gaps = None if gap is None else np.array([gap(s) for s in predictors])
    if not np.all(np.isfinite(objectives)) or (gaps is not None and not np.all(np.isfinite(gaps))):
        raise RuntimeError("non-finite loss along the computed trajectory")
    return Trajectory(times.copy(), covariances, predictors, objectives, gaps, factors, phase)


def _integrate(initial, times, rhs, *, method, rtol, atol):
    times = time_grid(times)
    scalar(rtol, "rtol")
    scalar(atol, "atol")
    result = solve_ivp(lambda _t, flat: rhs(flat.reshape(initial.shape)).ravel(),
                       (0.0, float(times[-1])), initial.ravel(), t_eval=times,
                       method=method, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    states = result.y.T.reshape(len(times), *initial.shape)
    if not np.all(np.isfinite(states)):
        raise RuntimeError("ODE integration produced non-finite states")
    return times, states


def integrate_covariance(initial: ArrayLike, loss: PredictorLoss, times: ArrayLike, *,
                         method: str = "DOP853", rtol: float = 1e-9, atol: float = 1e-11) -> Trajectory:
    """Solve the closed Bures ODE from a PSD covariance (singular is allowed)."""
    m = sum(loss.shape)
    a = matrix(initial, "initial covariance", (m, m))
    psd_eigh(a)
    t, states = _integrate((a + a.T)/2, times, lambda s: covariance_rhs(s, loss),
                           method=method, rtol=rtol, atol=atol)
    states = (states + states.transpose(0, 2, 1)) / 2
    return _result(t, states, loss)


def integrate_factors(initial: ArrayLike, loss: PredictorLoss, times: ArrayLike, *,
                      method: str = "DOP853", rtol: float = 1e-9, atol: float = 1e-11) -> Trajectory:
    """Integrate the actual factors, retaining every requested factor snapshot.

    Storage grows as len(times) * (d_u+d_v) * width. Paper experiments use a
    separate streaming output reducer for their large-width loss-only runs.
    """
    w = matrix(initial, "initial factors")
    factor_rhs(w, loss)
    t, states = _integrate(w, times, lambda s: factor_rhs(s, loss), method=method, rtol=rtol, atol=atol)
    covariances = states @ states.transpose(0, 2, 1) / w.shape[1]
    return _result(t, covariances, loss, factors=states)


def gradient_descent(initial: ArrayLike, loss: PredictorLoss, step_size: float, steps: int) -> Trajectory:
    """True GD on factors, including the initial state; no convergence is assumed."""
    tau = scalar(step_size, "step_size")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    w = matrix(initial, "initial factors").copy()
    factor_rhs(w, loss)
    factors = np.empty((steps + 1, *w.shape))
    factors[0] = w
    for k in range(steps):
        w = factor_step(w, loss, tau)
        if not np.all(np.isfinite(w)):
            raise RuntimeError(f"non-finite factor iterate at step {k+1}; reduce step_size")
        factors[k+1] = w
    covariances = factors @ factors.transpose(0, 2, 1) / w.shape[1]
    return _result(np.arange(steps + 1)*tau, covariances, loss, factors=factors)


def integrate_momentum(initial: ArrayLike, loss: PredictorLoss, times: ArrayLike, damping: float, *,
                       method: str = "DOP853", rtol: float = 1e-9, atol: float = 1e-11) -> Trajectory:
    """Integrate full phase closure; damping=0 is allowed but not a decay guarantee."""
    gamma = scalar(damping, "damping", allow_zero=True)
    m = sum(loss.shape)
    a = matrix(initial, "initial phase covariance", (2*m, 2*m))
    psd_eigh(a)
    t, states = _integrate((a+a.T)/2, times, lambda s: phase_covariance_rhs(s, loss, gamma),
                           method=method, rtol=rtol, atol=atol)
    states = (states + states.transpose(0, 2, 1))/2
    return _result(t, states[:, :m, :m].copy(), loss, phase=states)

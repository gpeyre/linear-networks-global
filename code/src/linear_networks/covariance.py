"""Uncentered second moments, signed spectral laws, and reproducible sampling."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, matrix, positive_integer, psd_eigh, scalar


def factor_covariance(factors: ArrayLike) -> FloatArray:
    """Return W W.T / n; columns are neurons, ordered as W = (U; V)."""
    w = matrix(factors, "factors")
    return w @ w.T / w.shape[1]


def signature_matrix(d_u: int, d_v: int) -> FloatArray:
    """Return J = diag(I_du, -I_dv)."""
    return np.diag(np.r_[np.ones(positive_integer(d_u, "d_u")),
                         -np.ones(positive_integer(d_v, "d_v"))])


def psd_square_root(covariance: ArrayLike, *, rtol: float = 1e-12) -> FloatArray:
    """Symmetric PSD square root, accepting singular covariances."""
    values, vectors = psd_eigh(covariance, rtol=rtol)
    return (vectors * np.sqrt(values)) @ vectors.T


def covariance_blocks(covariance: ArrayLike, d_u: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return (Sigma^u, Sigma^v, Sigma^vu), as views of the input when possible."""
    a = matrix(covariance, "covariance")
    d_u = positive_integer(d_u, "d_u")
    if a.shape[0] != a.shape[1] or d_u >= a.shape[0]:
        raise ValueError("covariance must be square with 0 < d_u < its size")
    return a[:d_u, :d_u], a[d_u:, d_u:], a[d_u:, :d_u]


def signed_spectrum(covariance: ArrayLike, d_u: int, *, rtol: float = 1e-12) -> FloatArray:
    """Sorted spectrum of Sigma^(1/2) J Sigma^(1/2), including every zero."""
    covariance_blocks(covariance, d_u)
    root = psd_square_root(covariance, rtol=rtol)
    pencil = root @ signature_matrix(d_u, len(root) - d_u) @ root
    return np.linalg.eigvalsh((pencil + pencil.T) / 2)


@dataclass(frozen=True)
class SupportMargins:
    """Supporting pencil margins a_plus, a_minus and their sum g(Sigma)."""
    a_plus: float
    a_minus: float

    @property
    def gap(self) -> float:
        return self.a_plus + self.a_minus


def support_margins(covariance: ArrayLike, d_u: int, *, rtol: float = 1e-12) -> SupportMargins:
    """Compute indexed margins, not the smallest nonzero eigenvalue.

    Eigenvalues within ``rtol * norm(covariance, 2)`` are treated as zero.
    Near a rank change this is a numerical diagnostic, not a rigorous certificate.
    """
    values = signed_spectrum(covariance, d_u, rtol=rtol)
    values[np.abs(values) <= rtol * np.linalg.norm(covariance, 2)] = 0
    d_v = len(values) - d_u
    return SupportMargins(float(max(0, values[d_v])), float(max(0, -values[d_v - 1])))


def sample_factors(covariance: ArrayLike, width: int, *, seed: int | None = None) -> FloatArray:
    """Sample independent zero-mean Gaussian columns with population covariance.

    The empirical covariance fluctuates; it is not forced to equal the population
    covariance. A local generator avoids changing NumPy's global random state.
    """
    width = positive_integer(width, "width")
    root = psd_square_root(covariance)
    return root @ np.random.default_rng(seed).standard_normal((len(root), width))


def white_phase_covariance(d_u: int, d_v: int, sigma_vu: float, sigma_qp: float = 0) -> FloatArray:
    """diag(sigma_vu^2 I_m, sigma_qp^2 I_m) in (u,v,p,q) order."""
    m = positive_integer(d_u, "d_u") + positive_integer(d_v, "d_v")
    position = scalar(sigma_vu, "sigma_vu", allow_zero=True)
    velocity = scalar(sigma_qp, "sigma_qp", allow_zero=True)
    return np.diag(np.r_[np.full(m, position**2), np.full(m, velocity**2)])

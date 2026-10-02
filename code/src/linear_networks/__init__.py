"""Training dynamics and covariance geometry of two-layer linear networks.

Importing this module does not import Matplotlib or change plotting backends.
"""
from .covariance import (
    SupportMargins, covariance_blocks, factor_covariance, psd_square_root,
    sample_factors, signature_matrix, signed_spectrum, support_margins,
    white_phase_covariance,
)
from .dynamics import covariance_rhs, covariance_step, factor_rhs, factor_step, force_matrix, phase_covariance_rhs
from .losses import PredictorLoss, QuadraticLoss, SmoothedLassoLoss
from .solvers import Trajectory, gradient_descent, integrate_covariance, integrate_factors, integrate_momentum

__version__ = "0.1.0"
__all__ = [
    "PredictorLoss", "QuadraticLoss", "SmoothedLassoLoss", "SupportMargins", "Trajectory",
    "covariance_blocks", "covariance_rhs", "covariance_step", "factor_covariance", "factor_rhs",
    "factor_step", "force_matrix", "gradient_descent", "integrate_covariance", "integrate_factors",
    "integrate_momentum", "phase_covariance_rhs", "psd_square_root", "sample_factors",
    "signature_matrix", "signed_spectrum", "support_margins", "white_phase_covariance",
]

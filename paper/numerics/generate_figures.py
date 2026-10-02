"""Reproduce the general-initialization figure and its numerical checks."""
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iclr-no-hi-mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def generator(error):
    dv, du = error.shape
    return np.block([[np.zeros((du, du)), error.T],
                     [error, np.zeros((dv, dv))]])


def margins(covariance, du):
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(values)) @ vectors.T
    signature = np.diag([1.0] * du + [-1.0] * (len(values) - du))
    signed = np.linalg.eigvalsh(root @ signature @ root)
    return signed[signed > 0].min(), -signed[signed < 0].max()


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9), constrained_layout=True)
    du = dv = 2
    covariance0 = np.diag([0.7, 2.2, 1.5, 3.1])
    target = np.full((dv, du), 0.05)
    gap0 = np.sum(target**2) / 2
    plus, minus = margins(covariance0, du)
    margin = plus + minus
    signature = np.diag([1., 1., -1., -1.])
    times = np.linspace(0, 3, 301)

    def rhs(t, flat):
        covariance = flat.reshape(4, 4)
        force = generator(covariance[du:, :du] - target)
        return (-force @ covariance - covariance @ force).ravel()

    flow = solve_ivp(rhs, (0, 3), covariance0.ravel(), t_eval=times,
                     method="DOP853", rtol=2e-11, atol=1e-13)
    assert flow.success
    gaps, law_error = [], 0.0
    law0 = [np.trace(np.linalg.matrix_power(signature @ covariance0, k))
            for k in range(1, 5)]
    for t, flat in zip(flow.t, flow.y.T):
        covariance = flat.reshape(4, 4)
        gap = np.sum((covariance[du:, :du] - target)**2) / 2
        assert gap <= gap0 * np.exp(-2 * margin * t) + 1e-12
        for k, expected in enumerate(law0, start=1):
            observed = np.trace(np.linalg.matrix_power(signature @ covariance, k))
            law_error = max(law_error, abs(observed - expected))
            assert_allclose(observed, expected, rtol=1e-8, atol=1e-8)
        gaps.append(gap)
    axes[0].semilogy(times, np.array(gaps) / gap0, color="#167d8d", label="Loss")
    axes[0].semilogy(times, np.exp(-2 * margin * times), "--",
                     color="#ba3c57", label="Certified bound")
    axes[0].set_title("(a) Covariance flow", loc="left")

    scale = margin / (2 * np.sqrt(2))
    length = np.sqrt(2 * gap0) / scale
    upper = np.linalg.norm(covariance0, 2) * np.exp(2 * length)
    tau = min(scale / (upper * np.sqrt(2 * gap0) + 4.5 * upper**2),
              (1 - 1 / np.sqrt(2)) * scale * np.exp(-2 * length) / gap0)
    contraction = 1 - 2 * scale * tau
    n = 4
    factors = np.sqrt(n) * np.linalg.cholesky(covariance0)
    fundamental = np.eye(4)
    gd_times, gd_gaps = [], []
    max_defect = 0.0
    steps = int(np.ceil(3 / tau))
    for k in range(steps + 1):
        covariance = factors @ factors.T / n
        error = covariance[du:, :du] - target
        gap = np.sum(error**2) / 2
        assert gap <= gap0 * contraction**k + 1e-12
        defect = np.linalg.norm(fundamental @ signature @ fundamental.T - signature, 2)
        max_defect = max(max_defect, defect)
        assert defect <= 1 - 1 / np.sqrt(2) + 1e-10
        assert np.linalg.norm(covariance, 2) <= upper + 1e-10
        gd_times.append(k * tau)
        gd_gaps.append(gap)
        update = np.eye(4) - tau * generator(error)
        new_factors = update @ factors
        new_covariance = new_factors @ new_factors.T / n
        assert_allclose(new_covariance, update @ covariance @ update.T, atol=1e-12)
        next_gap = np.sum((new_covariance[du:, :du] - target)**2) / 2
        assert next_gap <= gap - scale * tau * np.sum(error**2) + 1e-12
        factors = new_factors
        fundamental = update @ fundamental
    axes[1].semilogy(gd_times, np.array(gd_gaps) / gap0, color="#167d8d")
    axes[1].semilogy(gd_times, contraction**np.arange(steps + 1),
                     "--", color="#ba3c57")
    axes[1].set_title("(b) Factor GD", loc="left")

    position = np.diag([200., 240.])
    velocity = np.diag([2e-5, 0.])
    phase0 = np.block([[position, np.zeros((2, 2))],
                       [np.zeros((2, 2)), velocity]])
    teacher, damping, budget = 0.01, 1., 0.27
    energy0 = teacher**2 / 2 + np.trace(velocity) / 2
    amin, bmax = 200 * np.exp(-2 * budget), 240 * np.exp(2 * budget)
    rho = 4 * bmax + 2 * np.sqrt(2 * energy0)
    rate = amin * damping / (rho + 2 * damping**2)
    deformation = 4 / rate * np.sqrt(3 * energy0 / (2 * amin))
    assert deformation < budget

    def phase_rhs(t, flat):
        phase = flat.reshape(4, 4)
        error = phase[1:2, :1] - teacher
        drift = np.block([[np.zeros((2, 2)), np.eye(2)],
                          [-generator(error), -damping * np.eye(2)]])
        return (drift @ phase + phase @ drift.T).ravel()

    inertial = solve_ivp(phase_rhs, (0, 20), phase0.ravel(),
                         t_eval=np.linspace(0, 20, 2001), method="DOP853",
                         rtol=2e-11, atol=1e-13)
    assert inertial.success
    energies = []
    for t, flat in zip(inertial.t, inertial.y.T):
        phase = flat.reshape(4, 4)
        energy = (phase[1, 0] - teacher)**2 / 2 + np.trace(phase[2:, 2:]) / 2
        assert energy <= 3 * energy0 * np.exp(-rate * t) + 1e-11
        eigs = np.linalg.eigvalsh(phase[:2, :2])
        assert eigs[0] >= amin - 1e-8 and eigs[-1] <= bmax + 1e-8
        energies.append(energy)
    axes[2].semilogy(inertial.t, np.maximum(energies, 1e-20) / energy0,
                     color="#167d8d")
    axes[2].semilogy(inertial.t, 3 * np.exp(-rate * inertial.t),
                     "--", color="#ba3c57")
    axes[2].set_title("(c) Momentum", loc="left")
    for ax in axes:
        ax.set_xlabel("Mean-field time")
        ax.grid(alpha=0.16)
    axes[0].set_ylabel("Relative loss")
    axes[1].set_ylabel("Relative loss")
    axes[2].set_ylabel("Relative mechanical energy")
    axes[0].legend(frameon=False, fontsize=8)
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures/general_training.pdf")
    fig.savefig(ROOT / "figures/general_training.png", dpi=180)
    plt.close(fig)
    report = {
        "quadratic_loss_L_kappa": [1, 1],
        "covariance_diagonal": covariance0.diagonal().tolist(),
        "target_entries": 0.05,
        "signed_margins": [float(plus), float(minus)],
        "flow_exponent": float(2 * margin),
        "maximum_spectral_law_error": float(law_error),
        "gd_stepsize": float(tau), "gd_steps": steps,
        "maximum_signature_defect": float(max_defect),
        "momentum_position_diagonal": position.diagonal().tolist(),
        "momentum_velocity_diagonal": velocity.diagonal().tolist(),
        "momentum_teacher": teacher, "damping": damping,
        "budget": budget, "energy0": energy0,
        "momentum_rate": float(rate),
        "momentum_deformation_bound": float(deformation),
        "all_assertions_passed": True,
    }
    (ROOT / "numerics/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

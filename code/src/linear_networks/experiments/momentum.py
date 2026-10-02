"""Figure 4: exact phase-covariance ODE and certified loss envelopes."""
import numpy as np
from scipy.integrate import solve_ivp
from .common import plt, save_data, save_figure, validate_arrays
from matplotlib.lines import Line2D
from .certificates import DU, DV, ENERGY, strict_tube, tightened_rate, quantities
from .. import QuadraticLoss, phase_covariance_rhs

STEM = "figure4_momentum"
M = DU + DV
SCALES = np.array([200., 500., 1000.])
DAMPING = np.array([30., 100., 300., 1000.])
TEACHER = np.sqrt(2 / min(DU, DV)) * np.eye(DV, DU)
DISPLAY_FLOOR = 1e-14
ENERGY_TOLERANCE = 1e-5
OBJECTIVE = QuadraticLoss(TEACHER)


def phase_rhs(_t, flat, gamma):
    return phase_covariance_rhs(flat.reshape(2*M, 2*M), OBJECTIVE, gamma).ravel()


def compute(output):
    # Match the original integration and output grid, including the unplotted
    # tail. Figure 4 displays only t <= 1; no rate is fitted to these samples.
    times = np.unique(np.concatenate([[0], np.geomspace(1e-8, 0.1, 1200),
                                      np.linspace(0.1, 10, 3001)]))
    shape = (len(SCALES), len(DAMPING))
    losses, energies = np.empty((*shape, len(times))), np.empty((*shape, len(times)))
    rates, qvalues, budgets, gaps = [np.empty(shape) for _ in range(4)]
    position_min, position_max, relative_psd_error = [np.empty(shape) for _ in range(3)]
    for i, scale in enumerate(SCALES):
        for j, gamma in enumerate(DAMPING):
            budget, gap = strict_tube(scale, gamma)
            a, b = scale**2 * np.exp(-2*budget), scale**2 * np.exp(2*budget)
            rate, q = tightened_rate(a, b, gamma)
            initial = np.diag([scale**2] * M + [1.] * M)
            solution = solve_ivp(phase_rhs, (0, 10), initial.ravel(), args=(gamma,),
                                 t_eval=times, method="DOP853", rtol=2e-9, atol=1e-11)
            if not solution.success:
                raise RuntimeError(solution.message)
            path = solution.y.T.reshape(-1, 2*M, 2*M)
            path = (path + path.transpose(0, 2, 1)) / 2
            residual = path[:, DU:M, :DU] - TEACHER
            losses[i, j] = 0.5 * np.sum(residual**2, axis=(1, 2))
            energies[i, j] = losses[i, j] + 0.5 * np.trace(path[:, M:, M:], axis1=1, axis2=2)
            spectrum = np.linalg.eigvalsh(path[:, :M, :M])
            position_min[i, j], position_max[i, j] = spectrum.min(), spectrum.max()
            relative_psd_error[i, j] = max(0., -np.linalg.eigvalsh(path).min()) / scale**2
            rates[i, j], qvalues[i, j], budgets[i, j], gaps[i, j] = rate, q, budget, gap
        print(f"  Figure 4: completed position scale {scale:g}", flush=True)
    data = dict(times=times, scales=SCALES, damping=DAMPING, losses=losses,
                energies=energies, rates=rates, qvalues=qvalues, budgets=budgets,
                gaps=gaps, position_min=position_min, position_max=position_max,
                relative_psd_error=relative_psd_error)
    diagnostics = validate(data)
    save_data(output, STEM, **data)
    return data, diagnostics


def validate(data):
    shape = (len(SCALES), len(DAMPING))
    times = np.unique(np.concatenate([[0], np.geomspace(1e-8, 0.1, 1200),
                                      np.linspace(0.1, 10, 3001)]))
    validate_arrays(data, {
        "times": times.shape, "scales": SCALES.shape, "damping": DAMPING.shape,
        "losses": (*shape, len(times)), "energies": (*shape, len(times)),
        **{key: shape for key in ("rates", "qvalues", "budgets", "gaps",
                                  "position_min", "position_max", "relative_psd_error")},
    })
    np.testing.assert_array_equal(data["times"], times)
    np.testing.assert_array_equal(data["scales"], SCALES)
    np.testing.assert_array_equal(data["damping"], DAMPING)
    assert np.min(data["losses"]) >= 0
    # The unconstrained covariance solver is not exactly PSD-preserving at
    # roundoff-level tails. Report the signed residual; do not clip it away.
    assert np.min(data["energies"]) >= -ENERGY_TOLERANCE
    bounds = ENERGY / (1 - data["qvalues"][:, :, None]) * np.exp(
        -data["rates"][:, :, None] * data["times"])
    violation = float(np.max(data["losses"] - bounds))
    energy_violation = float(np.max(data["energies"] - bounds))
    assert np.all(data["gaps"] < 0)
    assert np.all((data["qvalues"] > 0) & (data["qvalues"] < 1))
    assert violation < 1e-7 and energy_violation < ENERGY_TOLERANCE
    a = data["scales"][:, None]**2 * np.exp(-2 * data["budgets"])
    b = data["scales"][:, None]**2 * np.exp(2 * data["budgets"])
    for i, sigma in enumerate(SCALES):
        for j, gamma in enumerate(DAMPING):
            budget, gap = strict_tube(sigma, gamma)
            np.testing.assert_allclose([data["budgets"][i, j], data["gaps"][i, j]],
                                       [budget, gap], rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose([data["rates"][i, j], data["qvalues"][i, j]],
                                       tightened_rate(a[i, j], b[i, j], gamma),
                                       rtol=1e-12, atol=1e-13)
            assert quantities(data["budgets"][i, j], sigma, gamma)[0] < 0
    assert np.all(data["position_min"] >= a - 1e-6)
    assert np.all(data["position_max"] <= b + 1e-6)
    assert np.max(data["relative_psd_error"]) < 1e-8
    return {"trajectories": 12, "unclipped_loss_bound_violation": violation,
            "energy_bound_violation": energy_violation,
            "minimum_raw_mechanical_energy": float(np.min(data["energies"])),
            "mechanical_energy_absolute_tolerance": ENERGY_TOLERANCE,
            "largest_relative_phase_PSD_error": float(data["relative_psd_error"].max()),
            "largest_certificate_gap": float(data["gaps"].max()),
            "display_floor": DISPLAY_FLOOR,
            "note": "Roundoff-level tails do not establish exact PSD or asymptotic rates"}


def plot(data, output, formats):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.6875), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.19, top=0.88, wspace=0.12)
    colors = plt.get_cmap("magma")(np.linspace(0.15, 0.85, len(DAMPING)))
    for i, (ax, scale) in enumerate(zip(axes, data["scales"])):
        for j, gamma in enumerate(data["damping"]):
            bound = ENERGY / (1 - data["qvalues"][i, j]) * np.exp(
                -data["rates"][i, j] * data["times"])
            ax.plot(data["times"], np.maximum(data["losses"][i, j], DISPLAY_FLOOR),
                    color=colors[j], lw=1.15, alpha=0.95)
            ax.plot(data["times"], bound, color=colors[j], lw=1.25, ls=":")
        ax.set_title(rf"({'abc'[i]}) $\sigma_{{vu}}={scale:g}$")
        ax.set(xlabel=r"$t$", xlim=(-0.02, 1.02), ylim=(DISPLAY_FLOOR, 10), yscale="log")
        ax.grid(True, color="#E8E8E8", lw=0.6)
    axes[0].set_ylabel(r"$\ell(\Sigma^{vu}_t)-\ell_\star$", fontsize=6)
    handles = [Line2D([0], [0], color=color, lw=1.8, label=rf"${gamma:g}$")
               for gamma, color in zip(data["damping"], colors)]
    axes[0].legend(handles=handles, title=r"$\gamma$", title_fontsize=4.6,
                   loc="upper right", ncol=2, fontsize=4.6, borderpad=0.25,
                   labelspacing=0.2, columnspacing=0.55, handlelength=1.4)
    handles = [Line2D([0], [0], color="#555555", lw=1.5, ls=style, label=label)
               for style, label in [("-", "loss"), (":", "energy bound")]]
    axes[2].legend(handles=handles, loc="upper right", fontsize=4.6,
                   borderpad=0.25, labelspacing=0.2, handlelength=1.4)
    save_figure(fig, output, STEM, formats)

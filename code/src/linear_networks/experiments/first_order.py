"""Figure 2: covariance flow and actual finite-width factor ODEs."""
import numpy as np
from scipy.integrate import RK45, solve_ivp
from .common import COLORS, GRID, plt, save_figure, save_data, check_times, validate_arrays
from .. import QuadraticLoss, covariance_rhs, factor_rhs, force_matrix, support_margins

STEM = "figure2_training"
DU, DV = 16, 9
M = DU + DV
ALPHAS = np.array([0.1, 0.5, 1.0, 5.0])
PERTURBATIONS = np.linspace(0.0, 1.0, 9)
WIDTHS = np.array([32, 128, 512, 2048])
TRIALS = 5


def force(error):
    return force_matrix(error)


def loss(predictor, teacher):
    return 0.5 * np.sum((predictor - teacher)**2, axis=(-2, -1))


def covariance_flow(initial, teacher, times):
    check_times(times)
    objective = QuadraticLoss(teacher)
    def rhs(_t, flat):
        return covariance_rhs(flat.reshape(M, M), objective).ravel()

    result = solve_ivp(rhs, (0, float(times[-1])), initial.ravel(), t_eval=times,
                       method="RK45", rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    path = result.y.T.reshape(-1, M, M)
    return loss(path[:, DU:, :DU], teacher)


def factor_flow(width, trial, teacher, times):
    check_times(times)
    seed = 10000 + 1000 * width + trial
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((M, width))
    objective = QuadraticLoss(teacher)

    def rhs(_t, flat):
        return factor_rhs(flat.reshape(M, width), objective).ravel()

    solver = RK45(rhs, 0., initial.ravel(), float(times[-1]), rtol=1e-8, atol=1e-10)
    # Use the same adaptive steps and dense interpolation as solve_ivp, but
    # reduce each output batch to losses instead of retaining all factor paths.
    losses = np.empty(len(times))
    losses[0] = loss(initial[DU:] @ initial[:DU].T / width, teacher)
    cursor = 1
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError(message)
        end = np.searchsorted(times, solver.t, side="right")
        if end > cursor:
            batch = solver.dense_output()(times[cursor:end]).T
            for i, flat in enumerate(batch, start=cursor):
                w = flat.reshape(M, width)
                losses[i] = loss(w[DU:] @ w[:DU].T / width, teacher)
            cursor = end
    assert cursor == len(times)
    covariance0 = initial @ initial.T / width
    gap = support_margins(covariance0, DU).gap
    return losses, gap


def compute(output):
    teacher = (np.random.default_rng(0).standard_normal((DU, DV)) / np.sqrt(DU)).T
    times = np.linspace(0, 40, 801)
    finite_times = np.linspace(0, 40, 501)
    scale_loss = np.array([covariance_flow(a * np.eye(M), teacher, times)
                           for a in ALPHAS])
    perturb_loss = []
    for scale in PERTURBATIONS:
        initial = np.eye(M)
        initial[0, 0] = initial[DU, DU] = scale
        perturb_loss.append(covariance_flow(initial, teacher, times))
    reference_loss = covariance_flow(np.eye(M), teacher, finite_times)
    finite_loss = np.empty((len(WIDTHS), TRIALS, len(finite_times)))
    empirical_gaps = np.empty((len(WIDTHS), TRIALS))
    for i, width in enumerate(WIDTHS):
        for trial in range(TRIALS):
            finite_loss[i, trial], empirical_gaps[i, trial] = factor_flow(
                int(width), trial, teacher, finite_times)
        print(f"  Figure 2: completed width {width}, {TRIALS} trials", flush=True)
    data = dict(teacher=teacher, times=times, finite_times=finite_times,
                seeds=10000 + 1000 * WIDTHS[:, None] + np.arange(TRIALS),
                alphas=ALPHAS, perturbations=PERTURBATIONS, widths=WIDTHS,
                scale_loss=scale_loss, perturb_loss=np.array(perturb_loss),
                reference_loss=reference_loss, finite_loss=finite_loss,
                empirical_gaps=empirical_gaps)
    diagnostics = validate(data)
    save_data(output, STEM, **data)
    return data, diagnostics


def validate(data):
    validate_arrays(data, {
        "teacher": (DV, DU), "times": (801,), "finite_times": (501,),
        "alphas": ALPHAS.shape, "perturbations": PERTURBATIONS.shape,
        "widths": WIDTHS.shape, "seeds": (len(WIDTHS), TRIALS),
        "scale_loss": (len(ALPHAS), 801),
        "perturb_loss": (len(PERTURBATIONS), 801), "reference_loss": (501,),
        "finite_loss": (len(WIDTHS), TRIALS, 501),
        "empirical_gaps": (len(WIDTHS), TRIALS),
    })
    for name, expected in (("alphas", ALPHAS), ("perturbations", PERTURBATIONS),
                           ("widths", WIDTHS), ("times", np.linspace(0, 40, 801)),
                           ("finite_times", np.linspace(0, 40, 501)),
                           ("seeds", 10000 + 1000 * WIDTHS[:, None] + np.arange(TRIALS))):
        np.testing.assert_array_equal(data[name], expected)
    expected_teacher = (np.random.default_rng(0).standard_normal((DU, DV)) / np.sqrt(DU)).T
    np.testing.assert_array_equal(data["teacher"], expected_teacher)
    for name in ("scale_loss", "perturb_loss", "reference_loss", "finite_loss"):
        assert np.min(data[name]) >= 0
    delta = loss(np.zeros((DV, DU)), data["teacher"])
    bound = delta * np.exp(-4 * data["alphas"][:, None] * data["times"])
    violation = float(np.max(data["scale_loss"] - bound))
    assert violation < 1e-7
    perturb_bound = delta * np.exp(-4 * data["perturbations"][:, None] * data["times"])
    perturb_violation = float(np.max(data["perturb_loss"] - perturb_bound))
    assert perturb_violation < 1e-7
    assert np.max(data["reference_loss"] - delta * np.exp(-4 * data["finite_times"])) < 1e-7
    assert np.min(data["empirical_gaps"]) > 0
    finite_violation = 0.0
    for i in range(len(WIDTHS)):
        for j in range(TRIALS):
            curve = data["finite_loss"][i, j]
            bound = curve[0] * np.exp(-2 * data["empirical_gaps"][i, j]
                                     * data["finite_times"])
            finite_violation = max(finite_violation, float(np.max(curve - bound)))
    assert finite_violation < 1e-7
    return {"scale_loss_bound_violation": violation,
            "perturb_loss_bound_violation": perturb_violation,
            "finite_loss_bound_violation": finite_violation,
            "minimum_sampled_support_gap": float(data["empirical_gaps"].min()),
            "factor_ODE_trials": len(WIDTHS) * TRIALS,
            "maximum_loss_bound_tolerance": 1e-7}


def plot(data, output, formats):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.6875), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.88, wspace=0.12)
    times = data["times"]
    teacher_norm = np.linalg.norm(data["teacher"])
    for i, alpha in enumerate(data["alphas"]):
        curve = data["scale_loss"][i]
        color = COLORS[i]
        coarse = alpha * (alpha / (np.sqrt(M * alpha**2 + 8 * teacher_norm**2)
                                   + 2 * teacher_norm))**(M-1)
        axes[0].plot(times, curve, color=color, lw=1.25, label=rf"$\alpha={alpha:g}$")
        axes[0].plot(times, curve[0] * np.exp(-4 * alpha * times), "--",
                     color=color, lw=0.75, alpha=0.65)
        axes[0].plot(times, curve[0] * np.exp(-4 * coarse * times), ":",
                     color=color, lw=0.8, alpha=0.8)
    axes[0].set_title("(a) Initialization scale")
    axes[0].set_ylabel(r"$\ell_t-\ell_\star$")
    axes[0].legend(loc="upper right", ncol=2, fontsize=4.6, borderpad=0.25,
                   labelspacing=0.2, columnspacing=0.55, handlelength=1.4)
    colors = plt.get_cmap("magma")(np.linspace(0.15, 0.85, len(data["perturbations"])))
    for scale, curve, color in zip(data["perturbations"], data["perturb_loss"], colors):
        label = rf"$r={scale:g}$" if np.isclose(scale, [0, 0.5, 1]).any() else "_nolegend_"
        axes[1].plot(times, curve, color=color, lw=1.15, label=label)
    axes[1].plot(times, data["perturb_loss"][-1, 0] * np.exp(-4 * times),
                 "--", color="black", lw=0.8, alpha=0.7)
    axes[1].set_title("(b) Covariance conditioning")
    axes[1].legend(loc="upper right", fontsize=4.6, borderpad=0.25,
                   labelspacing=0.2, handlelength=1.4)
    times = data["finite_times"]
    axes[2].plot(times, data["reference_loss"], color="black", lw=1.3, label="mean field")
    for i, width in enumerate(data["widths"]):
        for curve in data["finite_loss"][i]:
            axes[2].plot(times, curve, color=COLORS[i], lw=0.6, alpha=0.12)
        axes[2].plot(times, data["finite_loss"][i].mean(axis=0), color=COLORS[i],
                     lw=1.15, label=rf"$n={width}$")
    axes[2].set_title("(c) Finite width")
    axes[2].legend(loc="upper right", fontsize=4.4, borderpad=0.25,
                   labelspacing=0.18, handlelength=1.3)
    positive = data["scale_loss"][data["scale_loss"] > 0]
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.grid(True, **GRID)
    axes[0].set_ylim(10**(np.floor(np.log10(positive.min())) - 1),
                     10**np.ceil(np.log10(positive.max())))
    save_figure(fig, output, STEM, formats)

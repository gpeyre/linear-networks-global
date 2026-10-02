"""Figure 6: smoothed-Lasso covariance and finite-width factor ODEs."""
import numpy as np
from scipy.integrate import RK45, solve_ivp
from .common import COLORS, GRID, plt, save_figure, save_data, check_times, validate_arrays
from .. import SmoothedLassoLoss, covariance_rhs, factor_rhs, force_matrix, support_margins
from ..losses import smooth_lasso_gap

STEM = "figure6_smooth_lasso"
DU, DV = 16, 9
M = DU + DV
ALPHAS = np.array([0.1, 0.5, 1.0, 5.0])
PERTURBATIONS = np.linspace(0.0, 1.0, 9)
WIDTHS = np.array([32, 128, 512, 2048])
TRIALS = 5
LASSO_LAMBDA = 0.1
LASSO_DELTA = 0.01
KAPPA = 1.0
SMOOTHNESS = 1 + LASSO_LAMBDA / LASSO_DELTA


def force(error):
    return force_matrix(error)


def objective(predictor, teacher):
    return np.sum(0.5 * (predictor - teacher)**2
                  + LASSO_LAMBDA * (np.hypot(predictor, LASSO_DELTA) - LASSO_DELTA),
                  axis=(-2, -1))


def gradient(predictor, teacher):
    return (predictor - teacher
            + LASSO_LAMBDA * predictor / np.hypot(predictor, LASSO_DELTA))


def minimizer(teacher):
    return SmoothedLassoLoss(teacher, LASSO_LAMBDA, LASSO_DELTA).minimizer


def loss(predictor, optimum):
    return smooth_lasso_gap(predictor, optimum, LASSO_LAMBDA, LASSO_DELTA)


def covariance_flow(initial, teacher, optimum, times):
    check_times(times)
    objective = SmoothedLassoLoss(teacher, LASSO_LAMBDA, LASSO_DELTA)
    def rhs(_t, flat):
        return covariance_rhs(flat.reshape(M, M), objective).ravel()

    result = solve_ivp(rhs, (0, float(times[-1])), initial.ravel(), t_eval=times,
                       method="RK45", rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    path = result.y.T.reshape(-1, M, M)
    return loss(path[:, DU:, :DU], optimum)


def factor_flow(width, trial, teacher, optimum, times):
    check_times(times)
    seed = 10000 + 1000 * width + trial
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((M, width))
    objective = SmoothedLassoLoss(teacher, LASSO_LAMBDA, LASSO_DELTA)

    def rhs(_t, flat):
        return factor_rhs(flat.reshape(M, width), objective).ravel()

    solver = RK45(rhs, 0., initial.ravel(), float(times[-1]), rtol=1e-8, atol=1e-10)
    # Use the same adaptive steps and dense interpolation as solve_ivp, but
    # reduce each output batch to losses instead of retaining all factor paths.
    losses = np.empty(len(times))
    losses[0] = loss(initial[DU:] @ initial[:DU].T / width, optimum)
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
                losses[i] = loss(w[DU:] @ w[:DU].T / width, optimum)
            cursor = end
    assert cursor == len(times)
    covariance0 = initial @ initial.T / width
    gap = support_margins(covariance0, DU).gap
    return losses, gap


def compute(output):
    teacher = (np.random.default_rng(0).standard_normal((DU, DV)) / np.sqrt(DU)).T
    optimum = minimizer(teacher)
    times = np.linspace(0, 40, 801)
    finite_times = np.linspace(0, 40, 501)
    scale_loss = np.array([covariance_flow(a * np.eye(M), teacher, optimum, times)
                           for a in ALPHAS])
    perturb_loss = []
    for scale in PERTURBATIONS:
        initial = np.eye(M)
        initial[0, 0] = initial[DU, DU] = scale
        perturb_loss.append(covariance_flow(initial, teacher, optimum, times))
    reference_loss = covariance_flow(np.eye(M), teacher, optimum, finite_times)
    finite_loss = np.empty((len(WIDTHS), TRIALS, len(finite_times)))
    empirical_gaps = np.empty((len(WIDTHS), TRIALS))
    for i, width in enumerate(WIDTHS):
        for trial in range(TRIALS):
            finite_loss[i, trial], empirical_gaps[i, trial] = factor_flow(
                int(width), trial, teacher, optimum, finite_times)
        print(f"  Figure 6: completed width {width}, {TRIALS} trials", flush=True)
    data = dict(teacher=teacher, optimum=optimum,
                lasso_lambda=np.array(LASSO_LAMBDA), lasso_delta=np.array(LASSO_DELTA),
                kappa=np.array(KAPPA), smoothness=np.array(SMOOTHNESS),
                seeds=10000 + 1000 * WIDTHS[:, None] + np.arange(TRIALS),
                times=times, finite_times=finite_times,
                alphas=ALPHAS, perturbations=PERTURBATIONS, widths=WIDTHS,
                scale_loss=scale_loss, perturb_loss=np.array(perturb_loss),
                reference_loss=reference_loss, finite_loss=finite_loss,
                empirical_gaps=empirical_gaps)
    diagnostics = validate(data)
    save_data(output, STEM, **data)
    return data, diagnostics


def validate(data):
    validate_arrays(data, {
        "teacher": (DV, DU), "optimum": (DV, DU),
        "lasso_lambda": (), "lasso_delta": (), "kappa": (), "smoothness": (),
        "times": (801,), "finite_times": (501,), "alphas": ALPHAS.shape,
        "perturbations": PERTURBATIONS.shape, "widths": WIDTHS.shape,
        "seeds": (len(WIDTHS), TRIALS), "scale_loss": (len(ALPHAS), 801),
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
    assert float(data["lasso_lambda"]) == LASSO_LAMBDA
    assert float(data["lasso_delta"]) == LASSO_DELTA
    assert float(data["kappa"]) == KAPPA
    assert float(data["smoothness"]) == SMOOTHNESS
    teacher, optimum = data["teacher"], data["optimum"]
    residual = float(np.linalg.norm(gradient(optimum, teacher)))
    assert residual < 1e-12
    delta = loss(np.zeros((DV, DU)), optimum)
    bound = delta * np.exp(-4 * KAPPA * data["alphas"][:, None] * data["times"])
    violation = float(np.max(data["scale_loss"] - bound))
    assert violation < 1e-7
    # For paired diagonal perturbations the support gap is 2r. The r=0
    # case is singular with zero gap, so it has no positive rate certificate.
    bound = delta * np.exp(-4 * KAPPA * data["perturbations"][:, None] * data["times"])
    perturb_violation = float(np.max(data["perturb_loss"] - bound))
    assert perturb_violation < 1e-7
    assert np.min(data["empirical_gaps"]) > 0
    finite_violation = 0.0
    for i in range(len(data["widths"])):
        for j in range(TRIALS):
            curve = data["finite_loss"][i, j]
            bound = curve[0] * np.exp(-2 * KAPPA * data["empirical_gaps"][i, j]
                                     * data["finite_times"])
            finite_violation = max(finite_violation, float(np.max(curve - bound)))
    assert finite_violation < 1e-7
    for name in ("scale_loss", "perturb_loss", "reference_loss", "finite_loss"):
        assert np.min(data[name]) >= 0
    assert np.max(data["reference_loss"] - delta * np.exp(
        -4 * KAPPA * data["finite_times"])) < 1e-7

    # Independent objective and derivative checks, away from cancellation.
    rng = np.random.default_rng(6)
    direct_error = gradient_error = 0.0
    for _ in range(12):
        predictor = rng.standard_normal((DV, DU)) / np.sqrt(DU)
        direct = objective(predictor, teacher) - objective(optimum, teacher)
        direct_error = max(direct_error, abs(float(loss(predictor, optimum) - direct)))
        direction = rng.standard_normal((DV, DU))
        direction /= np.linalg.norm(direction)
        h = 1e-5
        difference = (objective(predictor + h * direction, teacher)
                      - objective(predictor - h * direction, teacher)) / (2 * h)
        gradient_error = max(gradient_error, abs(float(
            difference - np.sum(gradient(predictor, teacher) * direction))))
        for epsilon in (1e-4, 1e-8, 1e-12):
            nearby = optimum + epsilon * direction
            squared = np.sum((nearby - optimum)**2)
            gap = loss(nearby, optimum)
            assert 0.5 * KAPPA * squared <= gap <= 0.5 * SMOOTHNESS * squared
    assert direct_error < 1e-11 and gradient_error < 1e-7
    return {"initial_loss_gap": float(delta), "minimizer_gradient_norm": residual,
            "stable_vs_direct_loss_error": direct_error,
            "gradient_finite_difference_error": gradient_error,
            "scale_loss_bound_violation": violation,
            "perturb_loss_bound_violation": perturb_violation,
            "finite_loss_bound_violation": finite_violation,
            "minimum_sampled_support_gap": float(data["empirical_gaps"].min()),
            "factor_ODE_trials": len(data["widths"]) * TRIALS,
            "maximum_loss_bound_tolerance": 1e-7,
            "note": "Raw positive tails are solver dependent, not asymptotic rate evidence"}


def plot(data, output, formats):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.6875), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.88, wspace=0.12)
    times = data["times"]
    # Strong convexity bounds the initial sublevel set by radius
    # ||S(0)|| + 2 sqrt(2 Delta(0)/kappa); here S(0)=0.
    radius = 2 * np.sqrt(2 * loss(np.zeros((DV, DU)), data["optimum"]) / KAPPA)
    for i, alpha in enumerate(data["alphas"]):
        curve = data["scale_loss"][i]
        color = COLORS[i]
        coarse = alpha * (alpha / (np.sqrt(M * alpha**2 + 2 * radius**2)
                                   + radius))**(M-1)
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
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("time")
        ax.grid(True, **GRID)
        ax.set_xlim(0, 40)
    # Match the appendix triptych: keep raw tails, with no display cutoff.
    axes[0].set_ylim(1e-23, 10.)
    save_figure(fig, output, STEM, formats)

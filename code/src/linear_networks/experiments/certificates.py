"""Figures 3 and 5 and the initialization-only certificate used by Figure 4."""
import numpy as np
from scipy.optimize import brentq
from .common import plt, GRID, save_figure
from matplotlib import colors
from ..certificates import white_bootstrap, tightened_momentum_rate

DU, DV = 3, 2
KAPPA = SMOOTHNESS = 1.0
DELTA = SIGMA_QP = 1.0
ENERGY = DELTA + (DU + DV) * SIGMA_QP**2 / 2
SCALES = np.geomspace(1.0, 1000.0, 25)
GAMMAS = np.geomspace(10.0, 10000.0, 500)


def quantities(s, sigma, gamma):
    """Dimension-free bootstrap gap and rate used in the article."""
    return white_bootstrap(s, sigma, gamma, kappa=KAPPA, smoothness=SMOOTHNESS, energy=ENERGY)


def strict_tube(sigma, gamma):
    # Move just inside the first feasible boundary, rather than using the
    # boundary itself, where the theorem's strict inequality is not satisfied.
    grid = np.unique(np.concatenate([np.geomspace(1e-10, 5, 8000),
                                      np.linspace(1e-4, 1.5, 5000)]))
    gap, _ = quantities(grid, sigma, gamma)
    crossings = np.flatnonzero((gap[:-1] >= 0) & (gap[1:] < 0))
    if len(crossings) == 0:
        raise ValueError("No admissible tube found on the search grid")
    i = crossings[0]
    boundary = brentq(lambda s: quantities(s, sigma, gamma)[0],
                      grid[i], grid[i+1], xtol=1e-14, rtol=1e-13)
    budget = (boundary + grid[i+1]) / 2
    gap, _ = quantities(budget, sigma, gamma)
    if not gap < 0:
        raise ValueError("Tube is not strictly admissible")
    return float(budget), float(gap)


def tightened_rate(a, b, gamma):
    return tightened_momentum_rate(a, b, gamma, kappa=KAPPA, smoothness=SMOOTHNESS, energy=ENERGY)


def color_mapping():
    cmap = colors.LinearSegmentedColormap.from_list(
        "scale_colors", plt.get_cmap("magma")(np.linspace(0.18, 0.92, 256)))
    return cmap, colors.LogNorm(vmin=SCALES.min(), vmax=SCALES.max())


def colorbar(fig, ax, cmap, norm):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_label(r"$\sigma_{vu}$")


def windows(output, formats):
    fig, ax = plt.subplots(figsize=(2.2, 1.65), constrained_layout=True)
    s = np.geomspace(1e-3, 2, 500)
    cmap, norm = color_mapping()
    boundaries = 0
    for sigma in SCALES:
        gap, _ = quantities(s[:, None], sigma, GAMMAS[None, :])
        if gap.min() <= 0 <= gap.max():
            ax.contour(GAMMAS, s, gap, levels=[0],
                       colors=[cmap(norm(sigma))], linewidths=1.25)
            boundaries += 1
    ax.set(xscale="log", yscale="log", xlabel=r"$\gamma$", ylabel=r"$s$",
           title=r"Certified boundary $\mathfrak{G}_s(\gamma)=0$")
    colorbar(fig, ax, cmap, norm)
    save_figure(fig, output, "figure3_damping_windows", formats)
    return {"plotted_boundaries": boundaries, "grid": [500, 500],
            "position_scales": 25, "energy": ENERGY}


def boundary_rates():
    s = np.unique(np.concatenate([np.geomspace(1e-10, 2, 1800),
                                   np.linspace(0.02, 1.2, 800)]))
    rates = np.full((len(SCALES), len(GAMMAS)), np.nan)
    budgets = np.full_like(rates, np.nan)
    for i, sigma in enumerate(SCALES):
        gap, rate = quantities(s[:, None], sigma, GAMMAS[None, :])
        valid = (gap < 0).any(axis=0)
        first = np.argmax(gap < 0, axis=0)
        columns = np.flatnonzero(valid)
        rates[i, columns] = rate[first[columns], columns]
        budgets[i, columns] = s[first[columns]]
    return rates, budgets


def rate_plot(output, formats):
    rates, budgets = boundary_rates()
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    cmap, norm = color_mapping()
    points, asymptotic = [], []
    for sigma, curve in zip(SCALES, rates):
        valid = np.isfinite(curve) & (curve > 0)
        if not valid.any():
            continue
        ax.plot(GAMMAS[valid], curve[valid], color=cmap(norm(sigma)), lw=1.9)
        last = np.flatnonzero(valid)[-1]
        if last < len(GAMMAS) - 1:
            points.append((GAMMAS[last], curve[last]))
            gamma = KAPPA * sigma**3 / (24 * np.e * np.sqrt(3 * ENERGY / 2))
            asymptotic.append((gamma, KAPPA * sigma**2 * np.exp(-2/3) / (2 * gamma)))
    if points:
        ax.scatter(*np.array(points).T, marker="o", color="white", edgecolor="black",
                   s=28, linewidths=1.1, label="last certified grid point", zorder=5)
        ax.scatter(*np.array(asymptotic).T, marker="x", color="#009E73",
                   s=50, linewidths=1, label="asymptotic upper edge", zorder=6)
    ax.plot(GAMMAS, KAPPA * GAMMAS / (4 * SMOOTHNESS), color="black", lw=1.8,
            ls=":", label=r"large-$\sigma_{vu}$ limit")
    colorbar(fig, ax, cmap, norm)
    ax.set(xscale="log", yscale="log", xlabel=r"$\gamma$",
           ylabel=r"$\lambda_{s_{\mathrm{cert}}}$",
           title=r"Boundary certificate rate vs. damping $\gamma$")
    ax.grid(True, **GRID)
    ax.legend(loc="best", fontsize=6)
    save_figure(fig, output, "figure5_boundary_rates", formats)
    assert np.all(budgets[np.isfinite(budgets)] > 0)
    return {"finite_grid_rates": int(np.isfinite(rates).sum()),
            "gamma_grid_points": len(GAMMAS), "position_scales": len(SCALES),
            "interpretation": "First strictly admissible sampled budget, not an optimized dynamical rate"}

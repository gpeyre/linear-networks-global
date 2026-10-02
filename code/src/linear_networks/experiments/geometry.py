"""Figure 1: scalar covariance-cone trajectories and invariant leaves."""
import numpy as np
from .common import plt, COLORS, save_figure

beta = 1.0
colors = COLORS

def rk4_theta(theta0, h, beta=1.0, t_final=4.0, n_steps=500):
    """Integrate theta' = 2(beta - h sinh(theta)) with a fixed-step RK4 scheme."""
    t = np.linspace(0.0, t_final, n_steps)
    theta = np.empty_like(t)
    theta[0] = theta0
    dt = t[1] - t[0]

    def f(th):
        return 2.0 * (beta - h * np.sinh(th))

    for k in range(n_steps - 1):
        th = theta[k]
        k1 = f(th)
        k2 = f(th + 0.5 * dt * k1)
        k3 = f(th + 0.5 * dt * k2)
        k4 = f(th + dt * k3)
        theta[k + 1] = th + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, theta


def trajectory(alpha, b, r0, beta=1.0, t_final=4.0, n_steps=500):
    """Return a W2/Bures trajectory with invariants b and alpha^2."""
    h = np.sqrt(alpha**2 + b**2)
    if h <= 0:
        raise ValueError("Use alpha^2+b^2 > 0 to avoid the cone vertex.")
    theta0 = np.arcsinh(r0 / h)
    t, theta = rk4_theta(theta0, h, beta=beta, t_final=t_final, n_steps=n_steps)
    r = h * np.sinh(theta)
    a = h * np.cosh(theta)
    b_arr = np.full_like(r, b)
    p = a + b_arr
    q = a - b_arr
    return {"t": t, "r": r, "a": a, "b": b_arr, "p": p, "q": q, "alpha": alpha, "h": h}


def add_curve_arrow(ax, x, y, color, frac=0.55):
    idx = int(frac * (len(x) - 2))
    ax.annotate(
        "",
        xy=(x[idx + 1], y[idx + 1]),
        xytext=(x[idx], y[idx]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25, shrinkA=0, shrinkB=0),
    )

def render(output, formats):
    fig = plt.figure(figsize=(2.15, 3.35))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], left=0.25, right=0.95, bottom=0.12, top=0.96, hspace=0.34)
    ax2 = fig.add_subplot(grid[0, 0])
    ax3 = fig.add_subplot(grid[1, 0], projection="3d")

    # -----------------------------------------------------------------------------
    # Panel A: equal variances, b=0.
    # -----------------------------------------------------------------------------
    r_min, r_max = -1.25, 1.45
    a_max = 1.85
    r_grid = np.linspace(r_min, r_max, 700)

    ax2.fill_between(r_grid, np.abs(r_grid), a_max, color="#EEF0F2", alpha=1.0, zorder=0)
    ax2.plot(r_grid, np.abs(r_grid), color="#727272", lw=1.0, label=r"cone boundary $s=|r|$")

    for alpha in [0.20, 0.35, 0.55, 0.80, 1.10, 1.40]:
        ax2.plot(
            r_grid,
            np.sqrt(r_grid**2 + alpha**2),
            color="#B7B7B7",
            lw=0.9,
            alpha=0.9,
            zorder=1,
        )

    ax2.plot([beta, beta], [abs(beta), a_max], color="#2B9C69", lw=2.0, label=r"global minima $r=\beta$")
    ax2.scatter([beta], [abs(beta)], s=24, color="#2B9C69", zorder=5)

    same_variance_specs = [
        (0.24, -1.05, colors[0]),
        (0.38, -0.92, colors[1]),
        (0.58, -0.76, colors[2]),
        (0.82, -0.58, colors[3]),
        (1.12, -0.34, colors[4]),
    ]
    for alpha, r0, color in same_variance_specs:
        tr = trajectory(alpha=alpha, b=0.0, r0=r0, beta=beta, t_final=4.2)
        ax2.plot(tr["r"], tr["a"], color=color, lw=2.15, zorder=3)
        ax2.scatter(tr["r"][0], tr["a"][0], color=color, s=24, marker="o", zorder=4)
        ax2.scatter(tr["r"][-1], tr["a"][-1], color=color, s=28, marker="s", zorder=4)
        add_curve_arrow(ax2, tr["r"], tr["a"], color, frac=0.48)

    ax2.text(-1.18, 1.70, r"leaves $s^2-r^2=\alpha^2$", color="#5F5F5F")
    ax2.text(beta + 0.035, 1.55, r"$r=\beta$", color="#207F52")
    ax2.set_xlim(r_min, r_max)
    ax2.set_ylim(0.0, a_max)
    ax2.set_xlabel(r"predictor $r$")
    ax2.set_ylabel(r"shared variance $s$")
    panel_a_title = "(a) Equal variances"
    ax2.set_title(panel_a_title, pad=2)
    ax2.set_aspect("auto")
    ax2.set_anchor("C")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(color="#FFFFFF", lw=1.0)

    # -----------------------------------------------------------------------------
    # Panel B: unequal variances, full Lorentz cone.
    # -----------------------------------------------------------------------------
    rho = np.linspace(0.0, 1.72, 58)
    phi = np.linspace(0.0, 2 * np.pi, 96)
    RHO, PHI = np.meshgrid(rho, phi)
    X_cone = RHO * np.cos(PHI)
    Y_cone = RHO * np.sin(PHI)
    Z_cone = RHO
    ax3.plot_surface(X_cone, Y_cone, Z_cone, color="#BFC5CA", alpha=0.16, linewidth=0, shade=False)

    # A few circular rims make the cone readable in PDF.
    for rr in [0.45, 0.90, 1.35, 1.70]:
        ph = np.linspace(0, 2 * np.pi, 240)
        ax3.plot(rr * np.cos(ph), rr * np.sin(ph), rr * np.ones_like(ph), color="#8F969C", lw=0.45, alpha=0.45)
    for ph0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        rr = np.linspace(0, 1.72, 80)
        ax3.plot(rr * np.cos(ph0), rr * np.sin(ph0), rr, color="#8F969C", lw=0.4, alpha=0.35)

    # Translucent determinant leaves alpha^2 = a^2-r^2-b^2.
    RR, BB = np.meshgrid(np.linspace(-1.18, 1.42, 60), np.linspace(-0.88, 0.88, 54))
    for alpha, color, opacity in [(0.35, "#8EA7D8", 0.13), (0.72, "#E8B26C", 0.12)]:
        AA = np.sqrt(RR**2 + BB**2 + alpha**2)
        ax3.plot_surface(RR, BB, AA, color=color, alpha=opacity, linewidth=0, shade=False)

    # Global minimizer sheet r=beta within the cone.
    B_sheet, U_sheet = np.meshgrid(np.linspace(-0.92, 0.92, 30), np.linspace(0.0, 0.72, 14))
    R_sheet = beta * np.ones_like(B_sheet)
    A_sheet = np.sqrt(beta**2 + B_sheet**2) + U_sheet
    ax3.plot_surface(R_sheet, B_sheet, A_sheet, color="#2B9C69", alpha=0.18, linewidth=0, shade=False)
    ax3.plot(beta * np.ones(200), np.linspace(-0.92, 0.92, 200), np.sqrt(beta**2 + np.linspace(-0.92, 0.92, 200)**2), color="#2B9C69", lw=1.8)

    unequal_specs = [
        (0.30, -0.64, -1.05, colors[0]),
        (0.46, -0.32, -0.92, colors[1]),
        (0.64, 0.00, -0.76, colors[2]),
        (0.46, 0.32, -0.92, colors[3]),
        (0.30, 0.64, -1.05, colors[4]),
    ]
    for alpha, b0, r0, color in unequal_specs:
        tr = trajectory(alpha=alpha, b=b0, r0=r0, beta=beta, t_final=4.25)
        ax3.plot(tr["r"], tr["b"], tr["a"], color=color, lw=2.3)
        ax3.scatter(tr["r"][0], tr["b"][0], tr["a"][0], color=color, s=24, marker="o", depthshade=False)
        ax3.scatter(tr["r"][-1], tr["b"][-1], tr["a"][-1], color=color, s=30, marker="s", depthshade=False)

    # One invariant plane to make b conservation explicit without clutter.
    b0 = unequal_specs[0][1]
    R_plane, A_plane = np.meshgrid(np.linspace(-1.18, 1.42, 16), np.linspace(0.15, 1.72, 16))
    B_plane = b0 * np.ones_like(R_plane)
    mask = A_plane >= np.sqrt(R_plane**2 + B_plane**2)
    R_plane = np.where(mask, R_plane, np.nan)
    A_plane = np.where(mask, A_plane, np.nan)
    B_plane = np.where(mask, B_plane, np.nan)
    ax3.plot_surface(R_plane, B_plane, A_plane, color="#2458A6", alpha=0.08, linewidth=0, shade=False)

    ax3.text(-1.10, -0.85, 1.98, r"cone $a^2\geq r^2+b^2$", color="#596067")
    ax3.text(beta - 0.9, 0.72, 1.63, r"minima $r=\beta$", color="#207F52")
    ax3.set_xlim(-1.22, 1.45)
    ax3.set_ylim(-0.92, 0.92)
    ax3.set_zlim(0.0, 1.82)
    ax3.set_xlabel(r"$r$", labelpad=-5)
    ax3.set_ylabel(r"$b$", labelpad=-4)
    # A 2D label avoids the 3D backend clipping the z label in tight PDF export.
    ax3.set_zlabel("")
    ax3.text2D(1.07, 0.55, r"$a$", transform=ax3.transAxes)
    ax3.tick_params(axis="both", which="major", labelsize=5, pad=-1)
    panel_b_title = "(b) Unequal variances"
    ax3.set_title(panel_b_title, pad=0)
    ax3.view_init(elev=23, azim=-58)
    ax3.set_box_aspect((1.25, 0.90, 0.82), zoom=1.13)
    ax3.set_anchor("C")
    pos3 = ax3.get_position()
    height3 = pos3.height * 1.13
    ax3.set_position([pos3.x0 - 0.12, pos3.y1 - height3 - 0.005, pos3.width * 1.13, height3])
    ax3.grid(False)
    for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
        axis.pane.set_alpha(0.0)
    error = 0.0
    for alpha, r0, _ in same_variance_specs:
        tr = trajectory(alpha, 0, r0, t_final=4.2)
        error = max(error, float(np.max(np.abs(tr["p"]*tr["q"]-tr["r"]**2-alpha**2))))
    for alpha, b0, r0, _ in unequal_specs:
        tr = trajectory(alpha, b0, r0, t_final=4.25)
        error = max(error, float(np.max(np.abs(tr["p"]*tr["q"]-tr["r"]**2-alpha**2))))
    assert error < 1e-12
    save_figure(fig, output, "figure1_covariance_cone", formats)
    return {"maximum_leaf_identity_error": error, "RK4_points": 500,
            "panel_a_final_time": 4.2, "panel_b_final_time": 4.25}

"""Plot exact scalar Gaussian conservation leaves and their training paths."""
from pathlib import Path
import os
import tempfile
import numpy as np
from scipy.integrate import solve_ivp
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "iclr-no-hi-mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TEACHER = 0.8
COLORS = ["#2869a6", "#b74662", "#1c947a", "#7756a2"]


def trajectory(r0, imbalance, determinant):
    def rhs(t, state):
        r = state[0]
        a = np.sqrt(r*r + imbalance*imbalance + determinant)
        return [-2*a*(r-TEACHER)]
    result = solve_ivp(rhs, (0, 12), [r0], t_eval=np.linspace(0, 12, 700),
                       method="DOP853", rtol=1e-11, atol=1e-13)
    assert result.success
    r = result.y[0]
    a = np.sqrt(r*r + imbalance*imbalance + determinant)
    assert abs(r[-1]-TEACHER) < 1e-7
    assert np.allclose(a*a-r*r-imbalance*imbalance, determinant)
    assert np.all(np.diff((r-TEACHER)**2) <= 1e-12)
    return r, a


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.4, 3.4))
    left = fig.add_axes([.075, .18, .40, .70])
    right = fig.add_axes([.55, .09, .37, .82], projection="3d")
    rgrid = np.linspace(-1.2, 1.2, 400)
    left.fill_between(rgrid, np.abs(rgrid), 1.85, color="#e4e8eb", alpha=0.6)
    left.plot(rgrid, np.abs(rgrid), color="#28343e", lw=1.3)
    left.axvline(TEACHER, color="#278465", alpha=0.4, lw=1.5)
    left.text(TEACHER+0.045, 1.68, r"$r=\beta$", color="#278465")
    for r0, det, color in zip([-1.05, -.91, -.72, -.52],
                              [.08, .25, .55, .95], COLORS):
        left.plot(rgrid, np.sqrt(rgrid*rgrid+det), color="#a8b2b9", lw=0.8)
        r, a = trajectory(r0, 0, det)
        left.plot(r, a, color=color, lw=2.5)
        left.plot(r[0], a[0], "o", color=color, ms=5)
        left.plot(TEACHER, np.sqrt(TEACHER**2+det), "s", color=color, ms=5)
        j = 45
        left.annotate("", xy=(r[j+8], a[j+8]), xytext=(r[j-8], a[j-8]),
                      arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})
    left.set(xlim=(-1.2, 1.2), ylim=(0, 1.85), xlabel=r"Predictor $r$",
             ylabel=r"Mean variance $a$", title=r"(a) Equal variances: $b=0$")
    left.spines[["top", "right"]].set_visible(False)
    left.grid(alpha=0.13)

    rr, bb = np.meshgrid(np.linspace(-1.2, 1.2, 30), np.linspace(-.75, .75, 26))
    right.plot_surface(rr, bb, np.sqrt(rr*rr+bb*bb), color="#bdc7cd",
                       alpha=0.18, linewidth=0, shade=False)
    sheet_b, sheet_a = np.meshgrid(np.linspace(-.75, .75, 18),
                                  np.linspace(TEACHER, 1.85, 22))
    sheet_a = np.where(sheet_a**2 >= TEACHER**2+sheet_b**2, sheet_a, np.nan)
    right.plot_surface(np.full_like(sheet_a, TEACHER), sheet_b, sheet_a,
                       color="#5aa187", alpha=0.12, linewidth=0, shade=False)
    for r0, b, det, color in zip([-1.05, -.91, -.72, -.52],
                                 [-.6, -.2, .2, .6],
                                 [.08, .25, .55, .95], COLORS):
        right.plot(rgrid, np.full_like(rgrid, b),
                   np.sqrt(rgrid*rgrid+b*b+det), color="#a8b2b9", lw=0.8)
        r, a = trajectory(r0, b, det)
        right.plot(r, np.full_like(r, b), a, color=color, lw=2.5)
        right.scatter([r0], [b], [a[0]], color=color, s=25, depthshade=False)
        right.scatter([TEACHER], [b], [np.sqrt(TEACHER**2+b*b+det)],
                      color=color, marker="s", s=25, depthshade=False)
    right.set(xlim=(-1.2, 1.2), ylim=(-.75, .75), zlim=(0, 1.85),
              xlabel=r"$r$", ylabel=r"$b$",
              zlabel=r"$a$", title="(b) Unequal variances")
    right.set_xticks([-1, 0, 1])
    right.set_yticks([-.5, 0, .5])
    right.set_zticks([0, 1])
    right.set_box_aspect((1.25, .85, 1.))
    right.view_init(elev=23, azim=-58)
    for axis in [right.xaxis, right.yaxis, right.zaxis]:
        axis.labelpad = 0
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.15)
    fig.savefig(ROOT / "figures/gaussian_w2_cone_geometry.pdf")
    fig.savefig(ROOT / "figures/gaussian_w2_cone_geometry.png", dpi=180)
    plt.close(fig)
    print("PASS Gaussian geometry: 8 positive-determinant trajectories, conserved leaves, optimal endpoints")


if __name__ == "__main__":
    main()

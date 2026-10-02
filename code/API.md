# Computational API

The public `linear_networks` package depends only on NumPy and SciPy. Importing
it does not select a Matplotlib backend, change random seeds, or create files.
Functions validate dimensions, finiteness, and relevant sign/PSD conditions;
invalid inputs raise `ValueError`. Failed integrations raise `RuntimeError`.

## Conventions

- Factors: `W.shape == (d_u + d_v, n)`, with `U = W[:d_u]`, `V = W[d_u:]`.
- Predictor: `S = V @ U.T / n`, shape `(d_v, d_u)`.
- Covariance: `Sigma = W @ W.T / n`, **not centered**.
- Momentum phase order: `(u, v, p, q)`, with `(p, q) = (u', v')`.
- Time: the factor ODE is the gradient flow of **`n * loss(S)`**.
  There is no extra width factor in `U' = -E.T @ V`, `V' = -E @ U`.
- `sigma_vu` and `sigma_qp` are standard deviations, not variances.
- All solver output grids start at zero and increase strictly.

## Losses

`QuadraticLoss(target, weights=None)` supports rectangular predictors and
nonnegative entrywise weights. Zero weights leave predictor entries unobserved,
so the minimizer need not be unique. `kappa` is the smallest positive weight,
`smoothness` the largest, and `minimum` is zero. For the constant all-zero loss,
both constants are reported as zero; positive-rate theorems do not apply.

`SmoothedLassoLoss(target, strength=0.1, smoothing=0.01)` implements the objective
of Appendix H. Its unique `.minimizer` is computed by entrywise root finding,
and `.gap(S)` uses a cancellation-resistant expression instead of subtracting
nearly equal objectives. Its constants are `kappa=1` and
`smoothness=1 + strength/smoothing`. Smoothing must be strictly positive.

Both losses expose `.shape`, `.value(S)`, `.gradient(S)`, and `.gap(S)`.
A custom loss only needs the first three:

```python
class MyLoss:
    shape = (1, 2)

    def value(self, S):
        return float(0.5 * ((S - 1.0)**2).sum())

    def gradient(self, S):
        return S - 1.0
```

An optional `gap(S)` supplies known loss gaps for trajectory output. The
package does not prove PL, convexity, smoothness, or global convergence for a
user-supplied loss. It does not infer a minimizer from the final iterate.

## Geometry

| Function | Return value |
| --- | --- |
| `factor_covariance(W)` | `W @ W.T / width` |
| `covariance_blocks(Sigma, d_u)` | `(Sigma_u, Sigma_v, Sigma_vu)` |
| `signature_matrix(d_u, d_v)` | `diag(I_du, -I_dv)` |
| `psd_square_root(Sigma)` | Symmetric PSD square root, including singular cases |
| `signed_spectrum(Sigma, d_u)` | Ordered spectrum of `sqrt(Sigma) @ J @ sqrt(Sigma)` |
| `support_margins(Sigma, d_u)` | Object with `.a_plus`, `.a_minus`, `.gap` |
| `sample_factors(Sigma, width, seed=...)` | Independent Gaussian neuron columns |
| `white_phase_covariance(d_u, d_v, sigma_vu, sigma_qp=0)` | Uncorrelated position/velocity covariance |

**Zeros matter.** With zero-based indexing and `d_v = len(Sigma)-d_u`, the
support margins are `spectrum[d_v]` and `-spectrum[d_v-1]`. Never drop zero
eigenvalues before computing these margins. Positive support gap is not the
same as full rank, nor the sum of minimum eigenvalues of the diagonal blocks.

The margin routine treats signed eigenvalues within `rtol * norm(Sigma, 2)`
as zero, with `rtol=1e-12` by default. Nearly singular inputs need a conditioning
and tolerance study, not an unconditional claim of a positive gap. PSD square
roots reject genuinely indefinite inputs and clip only accepted roundoff-level
negative eigenvalues. Sampling does not artificially whiten the empirical
covariance back to the population covariance.

## Dynamics and Solvers

| Routine | State and update |
| --- | --- |
| `factor_rhs(W, loss)` | First-order velocity of actual factors |
| `covariance_rhs(Sigma, loss)` | Closed Bures ODE |
| `factor_step(W, loss, step_size)` | Simultaneous updates of `U` and `V` |
| `covariance_step(Sigma, loss, step_size)` | Exact induced factor-GD covariance update |
| `phase_covariance_rhs(phase, loss, damping)` | Full second-order phase closure |
| `integrate_factors(W0, loss, times, ...)` | Adaptive factor ODE |
| `integrate_covariance(Sigma0, loss, times, ...)` | Adaptive covariance ODE |
| `gradient_descent(W0, loss, step_size, steps)` | Actual discrete factor GD |
| `integrate_momentum(phase0, loss, times, damping, ...)` | Adaptive momentum phase ODE |

Adaptive solvers accept `method`, `rtol`, and `atol`, passed to SciPy's
`solve_ivp`. Defaults are DOP853, `1e-9`, and `1e-11`. The figure protocols
override these when reproducing the original experiments. PSD initial states
may be singular. Zero momentum damping is supported computationally, without
claiming energy decay. GD takes a user-selected positive stepsize; it does not
silently replace it with a theoretical bound or perform line search.

Every high-level solver returns a `Trajectory` with `times`, `covariances`,
`predictors`, `objectives`, and `loss_gaps` (`None` without a loss `gap` method).
Factor methods also populate `factors`; momentum populates `phase_covariances`.
For momentum with a known loss gap, `.energies` returns that gap plus half the
velocity-covariance trace. GD returns `steps+1` snapshots, including initialization,
at effective times `k * step_size` for comparison with gradient flow.

Factor trajectory storage scales as `len(times) * (d_u+d_v) * width`. For the
large finite-width article experiments, the figure runner streams interpolated
factor states directly to losses instead of retaining the full trajectory.

The factor covariance update is

$$\Sigma_{k+1}=(I-\tau G_k)\Sigma_k(I-\tau G_k).$$

It contains `tau**2 * G @ Sigma @ G`. Euler stepping of the covariance ODE is
a different algorithm. Likewise, exact ODE conservation laws are only approximate
under an adaptive numerical integrator, and do not hold exactly under factor GD.
Computed covariance trajectories and momentum energies are not projected or
clipped; inspect signed PSD residuals and repeat with tighter tolerances when
tails approach floating-point error.

## Momentum Certificates

`linear_networks.certificates.white_bootstrap(s, sigma_vu, damping,
kappa=..., smoothness=..., energy=...)` returns the certificate gap and rate,
with broadcasting over budgets and dampings. Initial position covariance must
be `sigma_vu**2 * I`, and initial position/velocity cross covariance must vanish.
The energy is the initial predictor-loss gap plus half the velocity trace.

A **strictly negative** certificate gap proves the bootstrap condition under
the paper's hypotheses. A positive gap or unsuccessful grid search is only
inconclusive. It does not imply divergence. `tightened_momentum_rate(a, b,
damping, kappa=..., smoothness=..., energy=...)` gives `(rate, q)` for the
energy bound `energy / (1-q) * exp(-rate*t)` on an **already certified** tube.
It must not be used to assume that tube is valid.

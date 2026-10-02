# Experimental Specification

This document defines the experiments needed for the six article figures.
All arrays are double precision. Each random experiment uses its own NumPy
`default_rng` seed, so requesting a subset of figures does not change the
initializations. Plotting does not alter the arrays used for validation.

## Conventions

Neuron columns are ordered as $(u,v)$, with
$W=(U^\top,V^\top)^\top$, $m=d_u+d_v$, and
$\Sigma=WW^\top/n$. The predictor is $S=VU^\top/n=\Sigma^{vu}$,
of shape $d_v\times d_u$. For $E=\nabla\ell(S)$, set

$$G(E)=\begin{pmatrix}0&E^\top\\E&0\end{pmatrix}.$$

The mean-field-time equations used in Figures 2 and 6 are

$$\dot U=-E^\top V,\qquad \dot V=-EU,\qquad
\dot\Sigma=-G(E)\Sigma-\Sigma G(E).$$

In particular, the factor right-hand side has no additional $1/n$ outside
$E$: this is gradient flow for $n\ell(VU^\top/n)$. Finite-width experiments
integrate the actual factors, not their covariance closure. Streaming dense
interpolation reduces factor states to losses at each requested time, avoiding
storage of the full $m\times n$ trajectory.

## Figure 1: Covariance Geometry

The scalar predictor loss is $(r-1)^2/2$, and the covariance has entries
$(p,r;r,q)$. Write $a=(p+q)/2$, $b=(p-q)/2$. The PSD cone is
$a\ge\sqrt{r^2+b^2}$; $b$ and $\alpha^2=a^2-r^2-b^2$ are conserved.
For $h=\sqrt{\alpha^2+b^2}$, integrate

$$r=h\sinh\vartheta,\qquad a=h\cosh\vartheta,\qquad
\dot\vartheta=2(1-h\sinh\vartheta).$$

The hyperbolic parameterization keeps trajectories on their invariant leaves
up to rounding error. Classical RK4 uses 500 equally spaced output points
(499 steps), to time 4.2 in panel (a) and 4.25 in panel (b).

- Panel (a), shared variance ($b=0$): $(\alpha,r_0)$ equals
  $(0.24,-1.05)$, $(0.38,-0.92)$, $(0.58,-0.76)$,
  $(0.82,-0.58)$, $(1.12,-0.34)$.
- Panel (b): $(\alpha,b,r_0)$ equals $(0.30,-0.64,-1.05)$,
  $(0.46,-0.32,-0.92)$, $(0.64,0,-0.76)$,
  $(0.46,0.32,-0.92)$, $(0.30,0.64,-1.05)$.

The green line/sheet is the minimizer set $r=1$. In the first panel, the
shared variance is labeled $s$; it equals $a$ because $b=0$. Here $\alpha$
is a leaf parameter, not the initialization variance of Figures 2 and 6.
No Gaussian-distribution assumption is needed for this covariance geometry.

## Figures 2 and 6: First-Order Training

Both experiments use $(d_u,d_v)=(16,9)$ and the same synthetic teacher:
draw a $16\times9$ standard Gaussian matrix with seed 0, divide by $4$,
then transpose. Their protocols are otherwise identical:

| Panel | Initializations | Times and integrator |
| --- | --- | --- |
| (a) | $\Sigma_0=\alpha I_{25}$, $\alpha\in\{0.1,0.5,1,5\}$ | 801 points on $[0,40]$, RK45, `rtol=1e-9`, `atol=1e-11` |
| (b) | Start at $I_{25}$; replace entries $(0,0)$ and $(16,16)$ by $r\in\{0,1/8,\ldots,1\}$ | Same covariance solver and grid |
| (c) | I.i.d. standard Gaussian columns of $W_0$; $n=32,128,512,2048$; five trials per width | 501 points on $[0,40]$, factor RK45, `rtol=1e-8`, `atol=1e-10` |

Trial $j\in\{0,\ldots,4\}$ at width $n$ uses seed $10000+1000n+j$.
The black reference in panel (c) is the covariance flow from $I_{25}$.
Thin colored curves are individual trials and thick curves are arithmetic
means of their losses, not losses of averaged predictors. In these figures
$\alpha$ is a variance, not a standard deviation.

Figure 2 uses $\ell(S)=\|S-S_{\mathrm{teach}}\|_F^2/2$.
Figure 6 uses the strongly convex, smooth non-quadratic objective

$$\ell(S)=\tfrac12\|S-S_{\mathrm{teach}}\|_F^2+
\lambda\sum_{ij}(\sqrt{S_{ij}^2+\delta^2}-\delta),
\qquad \lambda=0.1,\quad\delta=0.01.$$

Its PL constant is $\kappa=1$ and its smoothness constant is
$L=1+\lambda/\delta=11$. The regularized minimizer is generally not the
teacher. Each of its entries is the unique zero of
$x-y+\lambda x/\sqrt{x^2+\delta^2}$ between 0 and teacher entry $y$.
Brent's method uses relative tolerance $4\epsilon_{\mathrm{machine}}$ and
absolute tolerance `nextafter(0, 1)`.

To avoid subtracting nearly equal objective values, Figure 6 evaluates loss
gaps via the rationalized Bregman expression. For predictor entry $x$ and
minimizer entry $z$, let $h_x=\sqrt{x^2+\delta^2}$ and
$h_z=\sqrt{z^2+\delta^2}$. The entry's gap contribution is

$$\tfrac12(x-z)^2+
\lambda\frac{\delta^2(x-z)^2}{h_z(h_xh_z+xz+\delta^2)}.$$

Dashed curves in panel (a) are $\Delta_0e^{-4\kappa\alpha t}$. Dotted
curves use $\Delta_0e^{-4\kappa c_\alpha(M)t}$, with

$$c_\alpha(M)=\alpha\left(
\frac{\alpha}{\sqrt{m\alpha^2+2M^2}+M}\right)^{m-1}.$$

Figure 2 uses the exact initial-sublevel radius $M=2\|S_{\mathrm{teach}}\|_F$;
Figure 6 uses $M=2\sqrt{2\Delta_0/\kappa}$. The latter follows from strong
convexity and $S_0=0$. Panel (b) displays only the white-reference envelope;
validation checks each trajectory against its own support gap $2r$.
The $r=0$ initialization has zero support gap, so no positive exponential
certificate is claimed for it.

Each finite-width trial is checked against
$\Delta_0e^{-2\kappa g(\Sigma_0)t}$. The support gap is computed from
the ordered spectrum of $\Sigma_0^{1/2}J\Sigma_0^{1/2}$, where
$J=\mathrm{diag}(I_{d_u},-I_{d_v})$.
All sampled widths exceed $m$ in this experiment.

## Figures 3 and 5: Momentum Certificates

Both use $(d_u,d_v)=(3,2)$, $\kappa=L=1$, $\Delta_0=1$, and velocity
standard deviation $\sigma_{qp}=1$. Thus the initial mechanical excess
energy is $\mathcal E_0=\Delta_0+m\sigma_{qp}^2/2=7/2$.
The position standard deviation has 25 logarithmic values on $[1,1000]$;
damping has 500 logarithmic values on $[10,10000]$.

For a deformation budget $s>0$, the dimension-free certificate is

$$a_s=\sigma_{vu}^2e^{-2s},\quad b_s=\sigma_{vu}^2e^{2s},\quad
\rho_s=4b_sL+2\sqrt{2L\mathcal E_0},\quad
\lambda_s=\frac{a_s\kappa\gamma}{\rho_s+2\gamma^2},$$

$$\mathfrak G_s(\gamma)=\frac4{\lambda_s}
\sqrt{\frac{3\mathcal E_0}{2a_s}}-s.$$

Figure 3 draws zero contours on 500 logarithmic budgets in $[10^{-3},2]$.
Negative values certify a strict bootstrap tube. An absent contour only means
no boundary was found on that finite grid; it is not evidence that training
fails or that a certificate cannot exist outside the grid.

Figure 5 selects the first strictly negative grid value using the union of
1800 logarithmic budgets on $[10^{-10},2]$ and 800 linear budgets on
$[0.02,1.2]$. It plots the resulting certificate rate. Circles mark the last
certified damping sample before an observed exit from feasibility. Crosses
show the asymptotic approximation

$$\gamma_{\max}\sim\frac{\kappa\sigma_{vu}^3}
{24e\sqrt{3\mathcal E_0/2}},\qquad
\lambda\sim\frac{\kappa\sigma_{vu}^2e^{-2/3}}{2\gamma_{\max}}.$$

The dotted line is $\kappa\gamma/(4L)$, the fixed-damping large-position-scale
limit. The curves approximate a feasibility-boundary rate: they are neither
fitted dynamical rates nor exact optimization over all continuous budgets.

## Figure 4: Momentum Trajectories

The particle equation is $\ddot W+\gamma\dot W=-G(E)W$, with the same
mean-field-time normalization as above. In phase ordering $(u,v,p,q)$,
where $(p,q)=(\dot u,\dot v)$, the covariance obeys

$$\dot\Sigma_{\mathrm{phase}}=D\Sigma_{\mathrm{phase}}+
\Sigma_{\mathrm{phase}}D^\top,\qquad
D=\begin{pmatrix}0&I_m\\-G(E)&-\gamma I_m\end{pmatrix}.$$

Use the dimensions and constants of Figure 3, a quadratic loss, and teacher
$\sqrt{2/\min(d_u,d_v)}\,I_{d_v\times d_u}$. The initial phase covariance
is $\mathrm{diag}(\sigma_{vu}^2I_5,I_5)$, with
$\sigma_{vu}\in\{200,500,1000\}$ and $\gamma\in\{30,100,300,1000\}$.

The full $10\times10$ covariance equation is integrated with DOP853,
`rtol=2e-9`, `atol=1e-11`, to time 10. Output times are the unique union
of 0, 1200 logarithmic times in $[10^{-8},0.1]$, and 3001 linear times
in $[0.1,10]$. The plot displays the early-time interval around $[0,1]$;
the unplotted tail is retained for validation.

Tube selection uses **the same dimension-free certificate as Figures 3 and 5**,
without an extra factor $\max(d_u,d_v)$. Search the union of 8000 logarithmic
budgets in $[10^{-10},5]$ and 5000 linear budgets in $[10^{-4},1.5]$.
Refine the first sign change with Brent's method (`xtol=1e-14`, `rtol=1e-13`),
then take the midpoint between the root and the first negative grid point.
The selected budget must satisfy the strict inequality.

On this certified tube, put $K=4a_s\kappa$, $H=4b_sL$, and

$$\eta=\frac{\gamma}{\gamma+\sqrt{\gamma^2+2\rho_s}},\quad
A=(1-\eta)K,\quad B=\rho_s+\frac{\gamma^2}{2\eta},\quad
\epsilon=\frac{2\gamma}{A+B},\quad q=\epsilon\sqrt{H/2}.$$

The plotted tightened rate is $\lambda=\epsilon A/(1+q)$, and the dotted
energy envelope is $\mathcal E_0e^{-\lambda t}/(1-q)$. The initialization's
zero position-velocity block makes the initial Lyapunov correction zero.
No observed eigenvalues or fitted trajectory rate are used to choose the
certificate. Observed eigenvalues are used only to verify its tube bounds.

## Validation and Numerical Limits

- First-order loss-envelope checks allow an absolute violation of `1e-7`.
  The computed maximum violations and sampled support gaps are reported.
- Momentum checks require strict certificate gaps and $0<q<1$, loss-envelope
  violation below `1e-7`, energy-envelope violation below `1e-5`, and tube
  eigenvalue bounds to absolute tolerance `1e-6`. The largest negative phase
  eigenvalue relative to $\sigma_{vu}^2$ must be below `1e-8`.
  The covariance integrator is not exactly PSD-preserving: its late-time
  mechanical energy can be slightly negative. The raw minimum is reported
  and must be at least `-1e-5`; it is not silently clipped to zero.
- Scalar leaf identities are checked to `1e-12`. Unit tests also compare RK4
  trajectories with an independent adaptive solve and factor trajectories
  with covariance trajectories from the same empirical initialization.
- Cached arrays must have the expected shapes, parameters, seeds, and time
  grids. They must be numeric and finite; object arrays and pickle loading
  are not allowed. Certificates are recomputed when cached results are read.

Figure 4 clips losses to $10^{-14}$ **only for plotting**. Its validation and
saved arrays use raw values. Figures 2 and 6 plot raw positive losses; Figure 6
has a lower axis limit of $10^{-23}$. Roundoff and adaptive-solver tolerances
can dominate these tails. The validation tolerances are numerical acceptance
criteria, not mathematical error bounds, and do not validate asymptotic
behavior below solver precision. Every trajectory is checked before its
figure is saved.

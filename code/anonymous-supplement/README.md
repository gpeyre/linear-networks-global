# Reproducing the Paper Figures

**Global Exponential Convergence of Two-Layer Linear Network Training**

Anonymous numerical supplementary material for the ICLR submission. This
source-only package reproduces all **six figures**, including the two appendix
figures, using synthetic experiments. It is independent of the manuscript
repository and requires no external datasets, credentials, GPU, Jupyter, or
LaTeX for the default PDF output. No precomputed results are bundled.

## Quick Start

Use **Python 3.11 or later**. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -B reproduce.py --check
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
`requirements-tested.txt` pins the tested dependencies. `requirements.txt`
provides wider compatible version ranges when the pinned versions are not
available. The verified environment was Python 3.14.3, NumPy 2.3.5, SciPy
1.17.1, Matplotlib 3.10.8, and threadpoolctl 3.6.0. Minor differences in solver
tails and typography can occur with other environments.

The command recomputes every trajectory, generates the six PDFs, runs the
focused unit tests, and validates the computed results. Validation failures
stop the run with a nonzero exit status. Allow a few minutes on a laptop CPU;
no accelerator is used. Computation is limited to one BLAS thread. Installing
dependencies may require internet access; reproduction itself does not.

## Figure Guide

| Figure | Content | Source | Generated PDF |
| --- | --- | --- | --- |
| 1 | Scalar covariance cone, invariant leaves, and trajectories | `geometry.py` | `figure1_covariance_cone.pdf` |
| 2 | Quadratic loss: initialization scale, conditioning, finite width | `first_order.py` | `figure2_training.pdf` |
| 3 | Momentum bootstrap admissibility boundaries | `certificates.py` | `figure3_damping_windows.pdf` |
| 4 | Momentum trajectories and certified energy envelopes | `momentum.py` | `figure4_momentum.pdf` |
| 5, appendix | Boundary certificate rates and large-scale asymptotics | `certificates.py` | `figure5_boundary_rates.pdf` |
| 6, appendix | Smoothed-Lasso counterpart of Figure 2 | `smooth_lasso.py` | `figure6_smooth_lasso.pdf` |

Figure numbering follows the article PDF. See [EXPERIMENTS.md](EXPERIMENTS.md)
for equations, parameters, random seeds, solver tolerances, and interpretation.
The experiments and panel ordering match the paper. Standalone PDFs use
Matplotlib's serif fonts; the paper embeds the plots in LaTeX. Some legend
wording and margins are clarified here. Pixel-identical typography is not the
reproduction target, and late-time solver noise is not meaningful agreement
or disagreement with a mathematical rate.

## Outputs and Commands

The default output directory is `outputs/`, relative to the working directory:

```text
outputs/
  figures/           Six PDF figures
  data/              Numeric arrays for Figures 2, 4, and 6
  verification.json  Parameters/diagnostics, environment, timings, and hashes
```

The report ends with `"status": "passed"` only after every requested figure
and validation completes. A rerun replaces outputs with the same names; use
a separate output directory to retain earlier runs. Hashes identify the files
from that run, not universal byte-identical outputs across library versions.

```sh
# Recompute a subset; Figures 1 and 3 are a quick installation smoke test.
python -B reproduce.py --figures 1 3
python -B reproduce.py --figures 2 6 --output another-run

# Generate PNG previews as well as PDFs.
python -B reproduce.py --formats pdf png

# Replot your own previously generated arrays, without rerunning the ODEs.
python -B reproduce.py --from-data outputs/data --output replotted

# Run only the fast consistency tests, without generating paper figures.
python -B -m unittest discover -s tests -v

# Optional LaTeX-native export; requires pdflatex and AMS font packages.
python -B reproduce.py --formats pgf
```

`--from-data` validates the arrays before plotting Figures 2, 4, and 6.
Figures 1, 3, and 5 are inexpensive and are recomputed in either mode. A full
reproduction is the default command, without `--from-data`. This package does
not read notebooks or fetch results from the original repository.

## Checks and Scope

The unit tests cover factor/covariance closure and trajectory agreement,
loss-gradient signs, covariance-cone invariants and integration accuracy,
smoothed-Lasso gradients and loss gaps, momentum energy dissipation, certificate
formulas and strict feasibility, and safe numeric data loading. Full-run
validation additionally checks all plotted ODE families against the relevant
loss or energy bounds, allowing explicitly reported numerical tolerances.

These are numerical consistency checks, not proofs. Figures 2, 4, and 6 solve
continuous-time ODEs with adaptive numerical integrators; they do not show
discrete gradient descent. No unrelated experiments or theory-audit scripts
are included. Do not use `python -O`, which disables assertions; the reproduction
entry point rejects it.

## Anonymous Source Archive

```sh
python -B package.py
```

This writes `supplementary.zip` next to this directory. The archive includes
only an explicit list of source files, these two documents, dependency lists,
tests, and a generated SHA-256 manifest. It excludes generated results,
notebooks, cached data, environments, repository history, and hidden system
files. Archive timestamps and permissions are normalized; PDFs generated by
the code omit author and creation-date metadata. No author names, affiliations,
personal URLs, or machine-specific paths are needed by this package.

The package builder checks for personal filesystem paths and email addresses;
these checks supplement, rather than replace, a manual anonymity review. The
allowlist prevents local outputs and caches from accidentally entering the
submission even after running the experiments.

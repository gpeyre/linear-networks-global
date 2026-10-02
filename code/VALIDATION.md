# Numerical Validation

The public toolbox was checked locally on 2026-09-27 using Python 3.14.3,
NumPy 2.3.5, SciPy 1.17.1, Matplotlib 3.10.8, and threadpoolctl 3.6.0.
The validation environment was temporary; no global Python installation was
modified. The Python 3.11/3.13 GitHub Actions workflow is configured but has
not yet run as part of this local check.

## Checks Completed

- Built both an editable installation and a regular wheel. Installed the wheel
  in a separate environment and ran it outside the repository, without
  `PYTHONPATH` or access to the anonymous supplementary scripts as dependencies.
- Passed all 29 unit tests: 17 public-API checks and 12 article-experiment checks.
- Executed all three example notebooks in fresh kernels with temporary working
  directories. Each uses only the installed public API and contains executable
  consistency checks. Source notebooks remain free of stored outputs.
- Recomputed all six article figures, including actual finite-width factor
  trajectories, and exported PDF and PNG files. Inspected all six plots and
  the three notebook plots visually.
- Rebuilt the arXiv PDF and source archive. Checked the clickable repository
  link in the contributions paragraph on page 3. The anonymous conference
  manuscript and submission supplement were not changed.

## Numerical Care

Support margins retain the signed spectrum's zero eigenvalues. A regression
test covers a rank-one covariance whose exact support gap is zero: using a
threshold relative only to its nearly zero signed spectrum can incorrectly
report a positive gap. The toolbox instead thresholds relative to the
covariance norm and documents this as a conditioning-dependent diagnostic.

Smoothed-Lasso gaps use a stationary-point Bregman expression instead of
subtracting nearly equal objective values. An additional stable branch handles
large opposite-sign entries; tests check both near-optimum and far-field cases.

Adaptive covariance integration is not an exactly PSD-preserving algorithm.
In the article's momentum experiment, the smallest raw mechanical excess
energy was approximately `-6.8e-7`, within the specified `1e-5` absolute
tolerance; the largest relative phase-PSD error was below `9e-12`. These
roundoff-level tails do not establish an asymptotic rate. Only display curves
are floored; validation uses raw arrays and reports signed errors.

For each new run, use its `verification.json`, which records the actual
versions, parameters, diagnostics, elapsed times, and output hashes. A past
successful run is not a substitute for checking a changed solver or parameter
set. Numerical agreement supports implementation correctness, not a proof of
the manuscript's theorems.

See [README.md](README.md) for the commands and
[API.md](API.md) for shape, normalization, and tolerance conventions.

## Directory Reorganization (2026-10-02)

After moving the public code, notebooks, figures, and anonymous snapshot into
`code/`, the installable wheel builds and imports outside the checkout. All
29 toolbox tests, 13 anonymous-supplement tests, and 15 manuscript check groups
pass. The six figure scripts recompute successfully, all nine notebooks are
valid notebook files, and the three maintained examples execute in fresh
kernels. The relocated conference manuscript compiles to 38 pages. Rebuilding
the anonymous source ZIP yields the same SHA-256 digest as before the move.

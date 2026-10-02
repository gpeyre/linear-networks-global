# Code and Figure Reproduction

The installable `linear_networks` toolbox, article experiments, and notebooks
live together in this directory. Start with [API.md](API.md) for the numerical
interface or [the example notebooks](notebooks/examples/README.md) for short
worked examples. The `notebooks/research/` directory preserves earlier
exploratory notebooks and their results.

Install the repository with `python -m pip install -e '.[figures]'`, then run:

```sh
linear-networks-figures --check --output outputs/paper
```

This recomputes all six figures using fixed synthetic initializations, records
parameters, solver checks, environment versions, timings, and file hashes, and
stops on validation failure. The command works outside the repository after
installation. It needs no notebook execution, data downloads, GPU, or LaTeX.
For the tested dependency versions, install `code/anonymous-supplement/requirements-tested.txt`
before installing the project. Compatible ranges are recorded in `pyproject.toml`.

| Article figure | Experiment module in `linear_networks.experiments` | Output PDF |
| --- | --- | --- |
| 1 | `geometry` | `figure1_covariance_cone.pdf` |
| 2 | `first_order` | `figure2_training.pdf` |
| 3 | `certificates` | `figure3_damping_windows.pdf` |
| 4 | `momentum` | `figure4_momentum.pdf` |
| 5 (appendix) | `certificates` | `figure5_boundary_rates.pdf` |
| 6 (appendix) | `smooth_lasso` | `figure6_smooth_lasso.pdf` |

The exact parameters, seeds, tolerances, and mathematical interpretations are
in [EXPERIMENTS.md](EXPERIMENTS.md). This is experimental reproduction, not a
promise of byte-identical curves at roundoff-level tails or identical TeX fonts.
The paper's existing figure assets are never overwritten by this command.

## Commands and Outputs

```sh
linear-networks-figures --figures 1 3 --output outputs/smoke
linear-networks-figures --formats pdf png --output outputs/paper
linear-networks-figures --from-data outputs/paper/data --output outputs/replot
python -m linear_networks.experiments.reproduce --figures 2 6 --output outputs/subset
```

Each output directory contains `figures/`, `data/` for Figures 2, 4, and 6,
and `verification.json`. Only a completed run has `"status": "passed"`.
Requested figure IDs and output formats are recorded; a subset is not reported
as a complete six-figure reproduction. Output directories are relative to the
working directory, and reruns replace files with the same names. Numeric cache
loading disallows pickles and checks shapes, finiteness, seeds, and parameters.

Use `--formats pgf` for LaTeX-native export, which additionally requires
pdfLaTeX and AMS font packages. All formats use the same computed arrays.
Figure production limits BLAS to one thread. The core toolbox never changes
process-wide thread settings or plotting configuration.

## Tests and Notebooks

```sh
python -m unittest discover -s code/tests -v
python code/execute_notebooks.py --output outputs/notebooks
```

The tests include rectangular dimensions, rank-deficient and zero-gap inputs,
factor/covariance closure, the discrete quadratic correction, signed-spectrum
conservation, actual factor/phase trajectory agreement, mechanical energy,
loss gradients, stable smoothed-Lasso gaps, error handling, and figure bounds.
Notebook execution starts fresh kernels in isolated temporary directories;
executed copies are written only to the chosen output directory.

The [anonymous ICLR package](anonymous-supplement/README.md) is a preserved
submission snapshot. From `code/anonymous-supplement/`, run
`python -B package.py` to refresh `code/supplementary.zip`. Its original
entry point still works independently. Public figure
experiments derive from that snapshot but now call the reusable toolbox for
their training vector fields, support margins, and momentum certificates.
They retain the original seeds, grids, methods, and tolerances. The public
smoothed-Lasso gap also handles opposite-sign large entries without cancellation
in the rationalized denominator. The two packages are not expected to have
identical file hashes or floating-point loss tails.

See [VALIDATION.md](VALIDATION.md) for the completed local checks and known
floating-point limitations.

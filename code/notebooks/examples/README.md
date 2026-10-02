# Illustrative Notebooks

These small examples teach the public API; they are not the scripts used for
the full-resolution article figures. Start from the repository root:

```sh
python -m pip install -e '.[notebooks]'
jupyter lab code/notebooks/examples
```

| Notebook | What it demonstrates |
| --- | --- |
| [covariance_flow.ipynb](covariance_flow.ipynb) | A scalar covariance leaf, exact factor/covariance closure, and the support-gap envelope |
| [factor_gradient_descent.ipynb](factor_gradient_descent.ipynb) | Sampled finite-width factors, true simultaneous GD, its quadratic covariance correction, and conservation drift |
| [momentum.ipynb](momentum.ipynb) | Zero initial speed, phase-covariance closure, loss versus mechanical energy, and initialization-only certificates |

Run all cells in order. No local filesystem assumptions, external datasets,
TeX installation, or notebook-specific helper modules are needed after installing
the package. Random examples use explicit local seeds. Only display curves are
floored for logarithmic plots; checks use the raw arrays. Each notebook should
take seconds on a CPU.

To execute every notebook reproducibly in a fresh kernel and save rendered
copies without altering these sources:

```sh
python code/execute_notebooks.py --output outputs/notebooks
```

Older notebooks are retained as development records in `../research/`.
Use [the figure runner](../../README.md) to reproduce all article figures.

# Supporting Material

This repository accompanies **Global Exponential Convergence of Two-Layer
Linear Network Training** by Stephen Zhang and Gabriel Peyré. The [paper PDF](paper/paper.pdf)
and its LaTeX sources are in `paper/`; `arxiv/` contains a standalone preprint.

## Citation

```bibtex
@misc{zhangpeyre2026global,
  author = {Zhang, Stephen and Peyr{\'e}, Gabriel},
  title = {Global Exponential Convergence of Two-Layer Linear Network Training},
  year = {2026},
  note = {Preprint}
}
```

## Code

The `code/` directory contains:

- `src/linear_networks/`: the installable NumPy/SciPy toolbox for factor and
  covariance flows, momentum, losses, and convergence certificates.
- `src/linear_networks/experiments/`: reproducible protocols for the paper figures.
- `notebooks/examples/`: short, runnable illustrations of the main dynamics.
- `notebooks/research/`: exploratory notebooks and notes.
- `figures/`: figure assets; `tests/`: numerical and API checks.
- `anonymous-supplement/`: a preserved submission supplement snapshot.

To install the toolbox (Python 3.11+) and reproduce the figures:

```sh
python -m pip install -e '.[figures,notebooks]'
linear-networks-figures --check --output outputs/paper
```

See the [code guide](code/README.md) for figure-by-figure instructions and
the [example notebooks](code/notebooks/examples/README.md) for a starting point.

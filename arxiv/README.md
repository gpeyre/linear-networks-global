# arXiv Manuscript

**Global Exponential Convergence of Two-Layer Linear Network Training**  
Stephen Zhang (`syz@syz.id.au`) and Gabriel Peyré (`gabriel.peyre@ens.fr`)

Author affiliations:

- Stephen Zhang: Flatiron Institute and NYU Center for Data Science, as listed
  on [his homepage](https://syz.id.au/).
- Gabriel Peyré: CNRS, ENS, PSL Univ., as listed on page 1 of
  [Token Sample Complexity of Attention](https://arxiv.org/pdf/2512.10656).
  Email verified against [his contact page](https://www.gpeyre.com/contact/).

This directory is self-contained. It includes the manuscript, appendices,
bibliography, local styles, and all six figures. No files elsewhere in the
repository, numerical reruns, network access, or shell escape are needed to
compile it with a standard TeX Live installation.

## Ready-to-Use Files

- `main.pdf`: compiled paper with named authors and acknowledgments.
- `arxiv-source.zip`: source archive to upload to arXiv; select `main.tex` and
  the pdfLaTeX engine if prompted.
- `main.tex`: entry point; section sources are in `sections/`.
- `references.bib` and `main.bbl`: editable and precompiled bibliography.
- `figures/`: original PGF artwork and its two required PNG components.

The upload archive deliberately excludes the compiled PDF, build logs, local
documentation, and unrelated repository files. Check the PDF produced by arXiv
before completing submission; this directory does not submit the paper.

## Build

From this directory, run:

```sh
make
```

Requirements: pdfLaTeX, BibTeX, latexmk, Make, and Python 3 (standard library
only, for ZIP packaging). TeX Live supplies the required LaTeX packages.
Build intermediates stay in `build/`; `main.pdf`, `main.bbl`, and
`arxiv-source.zip` are refreshed in this directory.

To compile the extracted upload archive without Make or Python, run:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

The bundled `main.bbl` makes BibTeX unnecessary for an unchanged archive.
After editing citations or the bibliography, run `bibtex main` after the first
LaTeX pass, then run LaTeX twice more.

Undefined-reference and citation warnings on the first pass are normal and
should disappear after the subsequent passes. Errors such as `Missing {
inserted` or `Undefined control sequence` are not normal; the commands above
stop immediately on errors instead of producing a partial PDF.

## Relationship to the Conference Version

This is a standalone preprint version of the manuscript in `paper/`, with
neutral typography, author information, and self-contained figure paths.
The PDF uses the standard 11-point LaTeX `article` class with Latin Modern
fonts and one-inch margins. It does not use a conference template, conference
branding, review line numbers, or a publication header. Pagination changes
with the layout.

Gabriel Peyré's funding acknowledgment follows page 10 of
[Token Sample Complexity of Attention](https://arxiv.org/pdf/2512.10656):
ERC project WOLF and France 2030 grant ANR-23-IACL-0008 (PRAIRIE-PSAI).
The other authors' funding and that paper's computing allocation are not
transferred here.

## Validation

The PDF was built with pdfLaTeX / TeX Live 2022. The source archive was
extracted into a separate temporary directory and compiled without BibTeX,
shell escape, or access to the rest of this repository. All references and
citations resolve. The author page, acknowledgments, and figures were checked
visually. This is a local validation, not a test on arXiv's submission server.

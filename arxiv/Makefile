LATEXMK ?= latexmk
PYTHON ?= python3

SOURCES := main.tex main.bbl references.bib $(wildcard *.sty) \
           $(wildcard sections/*.tex) $(wildcard figures/*.pgf) \
           $(wildcard figures/*.png)

.PHONY: all pdf source
all: pdf source

pdf:
	$(LATEXMK) -pdf -outdir=build -interaction=nonstopmode -halt-on-error -file-line-error -no-shell-escape main.tex
	cp build/main.pdf main.pdf
	cp build/main.bbl main.bbl

# Only compilation inputs belong in the upload, not the PDF or build logs.
source: pdf
	$(PYTHON) -c 'import sys, zipfile; z = zipfile.ZipFile("arxiv-source.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9); [z.write(p) for p in sorted(sys.argv[1:])]; z.close()' $(SOURCES)

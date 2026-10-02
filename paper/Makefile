PYTHON ?= python3

.PHONY: all check figures clean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

check:
	$(PYTHON) numerics/check_manuscript.py

figures:
	$(PYTHON) numerics/generate_figures.py
	$(PYTHON) numerics/generate_geometry.py

clean:
	latexmk -c paper.tex

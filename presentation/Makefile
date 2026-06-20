TEX = pdflatex
SRC = main.tex
PDF = main.pdf
FIGS = figures/frozen_headtohead.png figures/value_of_ssl.png figures/latent_space.png

all: $(PDF)

$(PDF): $(SRC) $(FIGS)
	$(TEX) -interaction=nonstopmode $(SRC)
	$(TEX) -interaction=nonstopmode $(SRC)

# Refresh the embedded figures from the latest result artifacts.
figs:
	cp ../results/benchmark/frozen_headtohead.png figures/
	cp ../results/label_eff/value_of_ssl.png figures/
	cp ../results/latent/latent_space.png figures/

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb build*.log
	rm -rf _qa

.PHONY: all figs clean

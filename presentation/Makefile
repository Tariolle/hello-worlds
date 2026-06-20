# Makefile -- EEG-JEPA presentation (Beamer / metropolis).
# Run from this directory: main.tex's \graphicspath uses ../results/..., so the
# figures resolve only when the build runs here.
#
#   make           build main.pdf with latexmk (handles all TikZ/TOC passes)
#   make manual    build without latexmk (two pdflatex passes)
#   make xe        build with xelatex (proper Fira fonts for metropolis)
#   make watch     rebuild on every save (latexmk -pvc)
#   make view      build, then open the PDF
#   make clean     remove aux files
#   make cleanall  also remove main.pdf

MAIN  := main
FLAGS := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all manual xe watch view clean cleanall

all:
	latexmk -pdf $(FLAGS) $(MAIN).tex

manual:
	pdflatex $(FLAGS) $(MAIN).tex
	pdflatex $(FLAGS) $(MAIN).tex

xe:
	latexmk -pdfxe $(FLAGS) $(MAIN).tex

watch:
	latexmk -pdf -pvc $(FLAGS) $(MAIN).tex

view: all
	-@open $(MAIN).pdf 2>/dev/null || xdg-open $(MAIN).pdf 2>/dev/null || start "" $(MAIN).pdf 2>/dev/null || true

clean:
	-latexmk -c $(MAIN).tex 2>/dev/null
	-rm -f $(MAIN).aux $(MAIN).log $(MAIN).nav $(MAIN).out $(MAIN).snm $(MAIN).toc $(MAIN).vrb $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).bbl $(MAIN).blg

cleanall: clean
	-rm -f $(MAIN).pdf

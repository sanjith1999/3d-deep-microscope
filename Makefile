.PHONY: all main supplement clean

all: main supplement

main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

supplement:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex

clean:
	latexmk -c main.tex supplement.tex

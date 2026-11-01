.PHONY: all main clean

all: main

main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex

clean:
	latexmk -c main.tex

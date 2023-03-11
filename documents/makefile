OPTS ?= -std=c99 -g -O2

.PHONY: all
all:
	echo test
	pdflatex idea.tex

.PHONY: check
check:
	echo Objects are idea.tex

.PHONY: clean
clean:
	-rm *.aux $(Exec)
	-rm *.log $(Exec)
	-rm *.pdf $(Exec)
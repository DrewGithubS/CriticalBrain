OPTS ?= -std=c99 -g -O2

.PHONY: all
all:
	echo test
	pdflatex Idea.tex

.PHONY: check
check:
	echo Objects are Idea.tex

.PHONY: clean
clean:
	-rm *.aux $(Exec)
	-rm *.log $(Exec)
	-rm *.pdf $(Exec)
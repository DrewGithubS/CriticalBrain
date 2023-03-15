OPTS ?= -std=c99 -g -O2

objects := $(sort $(wildcard *.tex))

.PHONY: documents
documents: $(objects)
	echo MAKING
	$(foreach object, $(objects), pdflatex $(object);)

.PHONY: check
check:
	echo Objects are $(objects)

.PHONY: clean
clean:
	-rm *.aux $(Exec)
	-rm *.log $(Exec)
	-rm *.pdf $(Exec)
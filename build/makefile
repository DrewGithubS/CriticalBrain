$(VERBOSE).SILENT:

OPTS ?= -std=c99 -g -O2
PATHTOBUILD=$(shell pwd)

subdirs := ../main/ ../documents
# $(sort $(dir $(wildcard */)))


.PHONY:
all: documents headers objects exec

.PHONY: headers
headers:
	if [ ! -d "./include" ]; then mkdir include; fi
	$(MAKE) -C ../main/ OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" headers

.PHONY: check
check: 
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" check)

.PHONY: clean
clean:
	rm include/*
	rm objects/*

.PHONY: objects
objects:
	if [ ! -d "./objects" ]; then mkdir objects; fi
	$(MAKE) -C ../main/ OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" objects

.PHONY: exec
exec:
	if [ ! -d "./exec" ]; then mkdir exec; fi
	$(MAKE) -C ../main/ OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" exec

.PHONY: documents
documents:
	$(MAKE) -C ../documents/ documents

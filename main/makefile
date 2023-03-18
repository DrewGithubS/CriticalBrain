$(VERBOSE).SILENT:

OPTS ?= -std=c99 -g -O2
PATHTOBUILD ?= ""

subdirs := $(filter-out exec/, $(sort $(dir $(wildcard */))))


.PHONY:
all:

.PHONY: check
check: 
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" check;)

.PHONY: headers
headers: 
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" headers;)

exec:
	$(MAKE) -C exec/ OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" all

.PHONY: objects
objects:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" all;)

.PHONY: exec
exec:
	$(MAKE) -C exec OPTS="$(OPTS)" PATHTOBUILD="$(PATHTOBUILD)" all;
	
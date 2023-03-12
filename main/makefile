OPTS ?= -std=c99 -g -O2

subdirs := $(sort $(dir $(wildcard */)))


.PHONY:
all:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" all;)

.PHONY: check
check: 
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" check;)

.PHONY: clean
clean:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" clean;)
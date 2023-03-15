OPTS ?= -std=c99 -g -O2

subdirs := build/
# $(sort $(dir $(wildcard */)))


.PHONY:
all:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" all;)

.PHONY: check
check:
	echo 
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" check;)

.PHONY: clean
clean:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" clean;)

.PHONY: objects
objects:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" objects;)

.PHONY: headers
headers:
	$(foreach subdir, $(subdirs), $(MAKE) -C $(subdir) OPTS="$(OPTS)" headers;)
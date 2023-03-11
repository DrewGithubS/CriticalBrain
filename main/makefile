Exec = main
OPTS ?= -std=c99 -g -O2

GCC=gcc

objects := $(patsubst %.c,%.o,$(wildcard *.c))


.PHONY: all
all: $(Exec)

.PHONY: check
check:
	echo Objects are $(objects)

$(objects): %.o: %.c *.h
	$(GCC) -c $(OPTS) $< -o $@

$(Exec): $(objects)
	$(GCC) $(OPTS) $(objects) -o $(Exec)

.PHONY: clean
clean:
	-rm *.o $(Exec)
Exec = main
OPTS ?= -std=c99 -g -O2

GCC=gcc

objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp))


.PHONY: all
all: $(Exec)

.PHONY: check
check:
	echo Objects are $(objects)

$(objects): %.o: %.cpp *.h
	$(GCC) -c $(OPTS) $< -o $@

$(Exec): $(objects) $(cu_objects)
	$(GCC) $(OPTS) $(objects) -o $(Exec)

.PHONY: clean
clean:
	-rm *.o $(Exec)
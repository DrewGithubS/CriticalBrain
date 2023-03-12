Exec = main

C_OPTS ?= -std=c99 -g -O2
CPP_OPTS ?= -std=c++11 -g -lSDL2
CU_OPTS ?= -std=c++11 -g -lSDL2 -O2

NVCC=nvcc

objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
cu_objects := $(patsubst %.cu,%.o,$(wildcard *.cu))


.PHONY: all
all: $(Exec)

.PHONY: check
check:
	echo Objects are $(objects)
	echo Cu_Objects are $(cu_objects)

$(objects): %.o: %.cpp *.h
	$(CXX) -c $(CPP_OPTS) $< -o $@

$(cu_objects): %.o: %.cu *.h
	$(NVCC) -c $(CU_OPTS) $< -o $@

$(Exec): $(objects) $(cu_objects)
	$(NVCC) $(OPTS) $(objects) $(cu_objects) -o $(Exec)

.PHONY: clean
clean:
	-rm *.o $(Exec)
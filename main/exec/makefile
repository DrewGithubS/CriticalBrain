Execs=main

PATHTOBUILD ?= ""

C_OPTS ?= -std=c99 -g -O2
CPP_OPTS ?= -std=c++11 -g -lSDL2
CU_OPTS ?= -std=c++11 -g -lSDL2 -O2

GXX=g++
NVCC=nvcc

objects := $(patsubst %.cpp,%,$(wildcard *.cpp))
cu_objects := $(patsubst %.cu,%,$(wildcard *.cu))


.PHONY: all
all: $(objects) $(cu_objects)

.PHONY: headers
headers:
	cp *.h $(PATHTOBUILD)/include


.PHONY: check
check:
	echo Objects are $(objects)
	echo Cu_Objects are $(cu_objects)

$(objects): %: %.cpp $(objects)
	$(NVCC) $< -I $(PATHTOBUILD)/include $(CPP_OPTS) $(PATHTOBUILD)/objects/* -o $(PATHTOBUILD)/exec/$@

$(cu_objects): %: %.cu $(cu_objects)
	$(NVCC) $< -I $(PATHTOBUILD)/include $(CU_OPTS) $(PATHTOBUILD)/objects/* -o $(PATHTOBUILD)/exec/$@
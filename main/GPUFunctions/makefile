PATHTOBUILD ?= ""

C_OPTS ?= -std=c99 -g -O2
CPP_OPTS ?= -std=c++11 -g -lSDL2
CU_OPTS ?= -std=c++11 -g -lSDL2 -O2

GXX=g++
NVCC=nvcc

objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
cu_objects := $(patsubst %.cu,%.o,$(wildcard *.cu))

all_objects := $(wildcard $(PATHTOOBJS)/*.o)


.PHONY: all
all:

.PHONY: headers
headers:
	cp *.h $(PATHTOBUILD)/include
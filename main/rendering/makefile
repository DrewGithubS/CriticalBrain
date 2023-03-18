$(VERBOSE).SILENT:

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
all: $(objects) $(cu_objects)

.PHONY: headers
headers:
	cp *.h $(PATHTOBUILD)/include


.PHONY: check
check:
	echo Objects are $(objects)
	echo Cu_Objects are $(cu_objects)

$(objects): %.o: %.cpp *.h
	$(GXX) -I $(PATHTOBUILD)/include -c $(CPP_OPTS) $< -o $(PATHTOBUILD)/objects/$@

$(cu_objects): %.o: %.cu *.h
	$(NVCC) -I $(PATHTOBUILD)/include -c $(CPP_OPTS) $< -o $(PATHTOBUILD)/objects/$@
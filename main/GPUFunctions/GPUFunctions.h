#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H

#include <cstdint>

void * gpuMemAlloc(uint32_t bytes);
void memcpyCPUtoGPU(void * to, void * from, size_t size);
void memcpyGPUtoCPU(void * to, void * from, size_t size);

#endif
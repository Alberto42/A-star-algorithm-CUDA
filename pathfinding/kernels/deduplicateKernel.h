//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_DEDUPLICATEKERNEL_H
#define PROJECT1B_DEDUPLICATEKERNEL_H

#include "../structures.h"
__global__ void deduplicateKernel(State *s, int *sSize, State *t, HashMapDeduplicate *h, int slidesCount);

void deduplicateKernelHost(State *devS, int *devSSize, State *devT, HashMapDeduplicate *devHD, int slidesCount);
#endif //PROJECT1B_DEDUPLICATEKERNEL_H

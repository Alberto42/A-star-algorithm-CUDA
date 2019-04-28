//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_EXPANDKERNEL_H
#define PROJECT1B_EXPANDKERNEL_H

#include "../structures.h"

__global__ void expandKernel(Vertex *start, Vertex *target, State *m, PriorityQueue *q, State *s, int *sSize,State
*qiCandidates, int *qiCandidatesCount, int slidesCount, int slidesCountSqrt);
#endif //PROJECT1B_EXPANDKERNEL_H

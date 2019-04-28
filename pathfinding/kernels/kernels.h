//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_KERNELS_H
#define PROJECT1B_KERNELS_H

#include "../structures.h"
#include "assert.h"
__global__ void improveMKernel(State *m, State *qiCandidates, int *qiCandidatesCount);


__global__ void checkIfTheEndKernel(State *m, PriorityQueue *q, int* result);
__global__ void checkExistanceOfNotEmptyQueue(PriorityQueue *q, int* isNotEmptyQueue);
bool checkExistanceOfNotEmptyQueueHost(PriorityQueue *devQ, int* devIsNotEmptyQueue);
bool checkIfTheEndKernelHost(State *devM, PriorityQueue *devQ,int *devIsTheEnd);

__global__ void createHashmapKernel(HashMap *h, Vertex *start, Vertex *target, int slidesCount, int slidesCountSqrt);
__global__ void getPathKernel(HashMap *h, State *m,Vertex *start, int slidesCount, Vertex* result, int *sizeResult);
#endif //PROJECT1B_KERNELS_H

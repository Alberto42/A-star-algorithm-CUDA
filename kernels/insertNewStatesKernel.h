//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_INSERTNEWSTATESKERNEL_H
#define PROJECT1B_INSERTNEWSTATESKERNEL_H

#include "../structures.h"

__global__ void insertNewStates(HashMap *h, State *t, int *sSize, PriorityQueue *q,Vertex *target, int slidesCount,
                                int slidesCountSqrt, int* end);
#endif //PROJECT1B_INSERTNEWSTATESKERNEL_H

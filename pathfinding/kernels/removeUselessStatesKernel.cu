//
// Created by albert on 27.04.19.
//

#include "removeUselessStatesKernel.h"

__global__ void removeUselessStates(HashMap *h, State *t, int *sSize, int slidesCount) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for (int i = id * MAX_S_SIZE; i < id * MAX_S_SIZE + sSize[id]; i++) {
        if (t[i].f == -1)
            continue;
        State *tmp = h->find(t[i].node, slidesCount);
        if (tmp->f != -1 && tmp->g < t[i].g)
            t[i].f = -1;
    }
}
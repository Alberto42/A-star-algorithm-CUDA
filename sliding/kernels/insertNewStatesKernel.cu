//
// Created by albert on 27.04.19.
//

#include "insertNewStatesKernel.h"
#include "assert.h"

__global__ void insertNewStates(HashMap *h, State *t, int *sSize, PriorityQueue *q, Vertex *target, int slidesCount,
                                int slidesCountSqrt, int *end) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;

    for (int i = id; i < THREADS_COUNT * MAX_S_SIZE; i += THREADS_COUNT) {
        if (t[i].f != -1) {

            t[i].f = t[i].g + f(t[i].node, *target, slidesCount, slidesCountSqrt);
            q[id].insert(t[i]);

            for (int j = 0; j < H_SIZE; j++) {

                int hash = t[i].node.hash(j, slidesCount);
                assert(0 <= hash && hash < H_SIZE);

                if (h->hashmap[hash].f == -1 || vertexEqual(h->hashmap[hash].node, t[i].node, slidesCount)) {

                    int lock = atomicExch(&h->hashmap[hash].lock, 0);
                    if (lock) {
                        h->hashmap[hash] = t[i];
                        int lock = atomicExch(&h->hashmap[hash].lock, 1);
                        assert(lock == 0);
                        break;
                    }
                }
                if (j == H_SIZE - 1) {
                    *end = 1;
                    return;
                }
            }
        }
    }
}

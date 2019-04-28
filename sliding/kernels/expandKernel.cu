
#include "expandKernel.h"
#include "../structures.h"
#include <assert.h>

__device__ __host__ bool vertexEqual(const Vertex &a, const Vertex &b,const int &slidesCount);

__device__ void expand(const State qi, State s[], int &sSize, const Vertex &target, int slidesCount, int
slidesCountSqrt) {
    int moves[] = {-1, 1, -slidesCountSqrt, slidesCountSqrt};
    const int movesCount = 4;
    int empty = -1;
    const int *slides = qi.node.slides;
    for (int i = 0; i < slidesCount; i++)
        if (slides[i] == 0) {
            empty = i;
            break;
        }
    if (empty == -1)
        assert(false);
    for (int i = 0; i < movesCount; i++) {
        int move = empty + moves[i];
        if (move < 0 || move >= slidesCount)
            continue;
        if (i < 2 && empty / slidesCountSqrt != move / slidesCountSqrt)
            continue;
        State sTmp;
        sTmp.g = qi.g + 1;

        sTmp.node = qi.node;
        swap(sTmp.node.slides[empty], sTmp.node.slides[move]);

        sTmp.f = -2;
        sTmp.prev = qi.node;
        sTmp.lock = 1;
        assert(sSize < MAX_S_SIZE);
        s[sSize++] = sTmp;
    }
}

__global__ void expandKernel(Vertex *start, Vertex *target, State *m, PriorityQueue *q, State *s, int *sSize, State
*qiCandidates, int *qiCandidatesCount, int slidesCount, int slidesCountSqrt) {

    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    sSize[id] = 0;
    for (int i = id * MAX_S_SIZE; i < (id + 1) * MAX_S_SIZE; i++)
        s[i].f = -1;
    if (q[id].empty()) {
        return;
    }
    State qi = q[id].pop();

    if (vertexEqual(qi.node, *target, slidesCount)) {
        if (qi.f < m->f) {
            int tmp = atomicAdd(qiCandidatesCount, 1);
            qiCandidates[tmp] = qi;
        }
    } else
        expand(qi, s + (id * MAX_S_SIZE), sSize[id], *target, slidesCount, slidesCountSqrt);
}

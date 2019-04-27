//
// Created by albert on 27.04.19.
//

#include "kernels.h"
__global__ void improveMKernel(State *m, State *qiCandidates, int *qiCandidatesCount) {
    for(int i=0;i<*qiCandidatesCount;i++) {
        if (qiCandidates[i].f < m->f) {
            *m = qiCandidates[i];
        }
    }
    *qiCandidatesCount = 0;
}


__global__ void checkIfTheEndKernel(State *m, PriorityQueue *q, int* result) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    State* t = q[id].top();
    if (t != nullptr) {
        if (m->f > t->f) {
            atomicExch(result, 0); //fixme: Maybe atomic is not necessary
        }
    }
}
__global__ void checkExistanceOfNotEmptyQueue(PriorityQueue *q, int* isNotEmptyQueue) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    if (!q[id].empty()) {
        atomicExch(isNotEmptyQueue, 1);
    }
}
bool checkExistanceOfNotEmptyQueueHost(PriorityQueue *devQ, int* devIsNotEmptyQueue) {
    int isNotEmptyQueue = 0;
    cudaMemcpy(devIsNotEmptyQueue, &isNotEmptyQueue, sizeof(int), cudaMemcpyHostToDevice);
    checkExistanceOfNotEmptyQueue<< < BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >> >(devQ, devIsNotEmptyQueue);
    cudaMemcpy(&isNotEmptyQueue, devIsNotEmptyQueue, sizeof(int), cudaMemcpyDeviceToHost);
    return isNotEmptyQueue;
}
bool checkIfTheEndKernelHost(State *devM, PriorityQueue *devQ,int *devIsTheEnd) {
    int isTheEnd = 1;

    cudaMemcpy(devIsTheEnd, &isTheEnd, sizeof(int), cudaMemcpyHostToDevice);

    checkIfTheEndKernel << < BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >> > (devM, devQ, devIsTheEnd);
    cudaMemcpy(&isTheEnd, devIsTheEnd, sizeof(int), cudaMemcpyDeviceToHost);

    return isTheEnd;
}

__global__ void createHashmapKernel(HashMap *h, Vertex *start, Vertex *target, int slidesCount, int slidesCountSqrt) {

    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for(int i=id;i<H_SIZE;i+=THREADS_COUNT) {
        h->hashmap[i].f = -1;
        h->hashmap[i].lock = 1;
    }
    if (id == 0) {
        State startState = State(0, f(*start, *target, slidesCount, slidesCountSqrt), *start);
        h->insert(startState, slidesCount);
    }
}
__global__ void getPathKernel(HashMap *h, State *m,Vertex *start, int slidesCount, Vertex* result, int *sizeResult) {
    State *currentState = m;
    while(true) {
        result[(*sizeResult)++] = currentState->node;
        if (vertexEqual(currentState->node, *start, slidesCount)) {
            break;
        }
        State* tmp = h->find(currentState->prev,slidesCount);
        assert(tmp->f != -1);
        currentState = tmp;
    }

}
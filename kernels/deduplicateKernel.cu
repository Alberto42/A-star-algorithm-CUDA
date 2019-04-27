//
// Created by albert on 27.04.19.
//

#include "deduplicateKernel.h"
__global__ void deduplicateKernel(State *s, int *sSize, State *t, HashMapDeduplicate *h, int slidesCount) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for(int i=id*MAX_S_SIZE;i<(id+1)*MAX_S_SIZE;i++) {
        t[i] = s[i];
    }
    for(int i=id*MAX_S_SIZE;i < id*MAX_S_SIZE + sSize[id];i++) {
        int z = h->find(s[i],slidesCount, s);
        int hash = s[i].hash(z, slidesCount);

        int t_tmp = atomicExch(h->hashmap+hash, i);
        if (t_tmp != -1 && vertexEqual(s[t_tmp].node, s[i].node, slidesCount) && s[t_tmp].g == s[i].g ) {
            t[i].f = -1;
        }
    }

}

void deduplicateKernelHost(State *devS, int *devSSize, State *devT, HashMapDeduplicate *devHD, int slidesCount) {

    HashMapDeduplicate hD;
    cudaMemcpy(devHD, &hD, sizeof(HashMapDeduplicate), cudaMemcpyHostToDevice);
    deduplicateKernel <<<BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT>>>(devS,devSSize, devT,devHD,slidesCount);
    cudaMemcpy(&hD, devHD, sizeof(HashMapDeduplicate), cudaMemcpyDeviceToHost); //fixme: useless ?

}
//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_REMOVEUSELESSSTATESKERNEL_H
#define PROJECT1B_REMOVEUSELESSSTATESKERNEL_H
#include "../structures.h"

__global__ void removeUselessStates(HashMap *h, State *t,int *sSize, int slidesCount);

#endif //PROJECT1B_REMOVEUSELESSSTATESKERNEL_H

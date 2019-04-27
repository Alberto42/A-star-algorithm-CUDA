//
// Created by albert on 27.04.19.
//

#include "structures.h"
#include <iostream>
#include <assert.h>


Vertex::Vertex(int slides[]) {
    memcpy(this->slides, slides, MAX_SLIDES_COUNT * sizeof(int));
}

__device__ __host__ Vertex::Vertex() {}
__device__ __host__ int Vertex::hashBase(int slidesCount, int base) {
    long int result = 0;
    for(long int i=0,p=1;i<slidesCount;i++,p=( p * base ) % H_SIZE) {
        result = (result + slides[i]*p) % H_SIZE;
    }
    return result;
}
__device__ __host__ int Vertex::hash1(int slidesCount) {
    return this->hashBase(slidesCount, 30);
}
__device__ __host__ int Vertex::hash2(int slidesCount) {
    int hash = this->hashBase(slidesCount, 29);
    hash = hash % 2 ? hash : (hash + 1) % H_SIZE;
    return hash;
}
__device__ __host__ int Vertex::hash(long int i, int slidesCount) {
    return (hash1(slidesCount) + i*hash2(slidesCount) ) % H_SIZE;
}
__host__ void Vertex::print(int slidesCount, ostream& out) {
    for(int i=0;i<slidesCount;i++){
        if (slides[i] == 0)
            out<<"_";
        else out<<slides[i];
        if (i != slidesCount-1)
            out<<",";
    }
    out << endl;
}


__device__ __host__ bool vertexEqual(const Vertex &a, const Vertex &b,const int &slidesCount) {
    for (int i = 0; i < slidesCount; i++) {
        if (a.slides[i] != b.slides[i])
            return false;
    }
    return true;
}



__device__ __host__ State::State():lock(1) {}

__device__ __host__ State::State(int f):f(f), lock(1) {}

__device__ __host__ State::State(int g, int f, Vertex node):g(g), f(f), node(node), lock(1) {}
__device__ __host__ State& State::operator=(const State& that) {
    this->node = that.node;
    this->g = that.g;
    this->f = that.f;
    this->prev = that.prev;
    return *this;
}

__device__ __host__ int State::hashBase(int slidesCount, int base) {
    int result = 0;
    int p=1;
    for(int i=0;i<slidesCount;i++,p=( p * base ) % H_SIZE_DEDUPLICATE) {
        result = (result + node.slides[i]*p) % H_SIZE_DEDUPLICATE;
    }
    result = (result + this->g*p) % H_SIZE_DEDUPLICATE;
    return result;
}
__device__ __host__ int State::hash1(int slidesCount) {
    return this->hashBase(slidesCount, 30);
}
__device__ __host__ int State::hash2(int slidesCount) {
    int hash = this->hashBase(slidesCount, 29);
    hash = hash % 2 ? hash : (hash + 1) % H_SIZE_DEDUPLICATE;
    return hash;
}
__device__ __host__ int State::hash(int i, int slidesCount) {
        return (hash1(slidesCount) + i*hash2(slidesCount) ) % H_SIZE_DEDUPLICATE;
    }

__device__ __host__ HashMap::HashMap() {
    for(int i=0;i<H_SIZE;i++)
        hashmap[i].f = -1;
}

__device__ __host__ State* HashMap::find(Vertex& v, int slidesCount) {
    for(int i=0;i<H_SIZE;i++) {
        int hash = v.hash(i,slidesCount);
        assert(0 <= hash && hash < H_SIZE);
        if (hashmap[hash].f == -1 || vertexEqual(hashmap[hash].node,v, slidesCount))
            return &hashmap[hash];
    }
    assert(false);
    return nullptr;
}

__device__ __host__ void HashMap::insert(State& s, int slidesCount) {
        State* tmp = this->find(s.node, slidesCount);
        *tmp = s;
}

HashMapDeduplicate::HashMapDeduplicate() {
        for(int i=0;i<H_SIZE_DEDUPLICATE;i++)
            hashmap[i] = -1;
}
__device__ int HashMapDeduplicate::find(State& item, int slidesCount, State* s) {
    for(int i=0;i<HASH_FUNCTIONS_COUNT;i++) {
        int hash = item.hash(i, slidesCount);
        assert(0 <= hash && hash < H_SIZE_DEDUPLICATE);
        if (hashmap[hash] == -1 || (vertexEqual(s[hashmap[hash]].node,item.node, slidesCount) && s[hashmap[hash]].g
                                                                                                 ==item.g) );
        return i;
    }
    return 0;
}


__device__ __host__ bool operator<(const State &a, const State &b) { return a.f < b.f; }

__device__ __host__ bool operator>(const State &a, const State &b) { return a.f > b.f; }

__device__ __host__ void swap(State &a, State &b) {
    State tmp = a;
    a = b;
    b = tmp;
}

__device__ __host__ void swap(int &a, int &b) {
    int tmp = a;
    a = b;
    b = tmp;
}

State A[PRIORITY_QUEUE_SIZE+1];
int lock;

__device__ __host__ PriorityQueue::PriorityQueue():lock(1) {}

int heapSize = 0;

__device__ __host__ int PriorityQueue::parent(int i) {
    return i / 2;
}

__device__ __host__ int PriorityQueue::left(int i) {
    return i * 2;
}

__device__ __host__ int PriorityQueue::right(int i) {
    return i * 2 + 1;
}

__device__ __host__ void PriorityQueue::maxHeapify(int i) {
    while(true) {
        int l = left(i);
        int r = right(i);
        int smallest;
        if (l <= heapSize && A[l] < A[i]) {
            smallest = l;
        } else {
            smallest = i;
        }
        if (r <= heapSize && A[r] < A[smallest])
            smallest = r;
        if (smallest != i) {
            swap(A[i], A[smallest]);
            i = smallest;
        } else {
            break;
        }
    }
}

__device__ __host__ void PriorityQueue::insert(State s) {
    assert(heapSize < PRIORITY_QUEUE_SIZE);
    heapSize++;
    A[heapSize] = s;
    int i = heapSize;
    while (i > 1 && A[parent(i)] > A[i]) {
        swap(A[i], A[parent(i)]);
        i = parent(i);
    }
}

__device__ State PriorityQueue::pop() {
    assert(heapSize > 0);
    State max = A[1];
    A[1] = A[heapSize];
    heapSize--;
    maxHeapify(1);
    return max;
}

__device__ bool PriorityQueue::empty() {
    return heapSize == 0;
}

__device__ State* PriorityQueue::top() {
    return (!this->empty()) ? A+1 : nullptr;
}
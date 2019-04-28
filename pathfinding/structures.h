//
// Created by albert on 27.04.19.
//

#ifndef PROJECT1B_STRUCTURES_H
#define PROJECT1B_STRUCTURES_H

#include <iostream>
#include <fstream>
using namespace std;
const int BLOCKS_COUNT = 41;
const int THREADS_PER_BLOCK_COUNT = 128;
const int THREADS_COUNT = BLOCKS_COUNT * THREADS_PER_BLOCK_COUNT;
const int MAX_SLIDES_COUNT = 25;
const int PRIORITY_QUEUE_SIZE = 10000;
const int MAX_S_SIZE = 4;
const int INF = 1000000000;
const int H_SIZE = 1048576*4; // It must be the power of 2
const int H_SIZE_DEDUPLICATE = 32768; // change to long ints
const int Q_CANDIDATES_COUNT = 100;
const int HASH_FUNCTIONS_COUNT = 32768; // must be smaller or equal to H_SIZE_DEDUPLICATE

struct Vertex {
    int slides[MAX_SLIDES_COUNT];
    __host__ __device__ Vertex(int slides[]);

    __device__ __host__ Vertex();
    __device__ __host__ int hashBase(int slidesCount, int base);
    __device__ __host__ int hash1(int slidesCount);
    __device__ __host__ int hash2(int slidesCount);
    __device__ __host__ int hash(long int i, int slidesCount);
    __host__ void print(int slidesCount, ostream& out);
};

__device__ __host__ bool vertexEqual(const Vertex &a, const Vertex &b,const int &slidesCount);

struct State {
    Vertex node;
    int g, f, lock;
    Vertex prev;

    __device__ __host__ State();

    __device__ __host__ State(int f);

    __device__ __host__ State(int g, int f, Vertex node);
    __device__ __host__ State& operator=(const State& that);

    __device__ __host__ int hashBase(int slidesCount, int base);
    __device__ __host__ int hash1(int slidesCount);
    __device__ __host__ int hash2(int slidesCount);
    __device__ __host__ int hash(int i, int slidesCount);
};

struct HashMap  {
    State hashmap[H_SIZE];
    __device__ __host__ HashMap();

    __device__ __host__ State* find(Vertex& v, int slidesCount);
    __device__ __host__ void insert(State& s, int slidesCount);
};
struct HashMapDeduplicate {
    int hashmap[H_SIZE_DEDUPLICATE];
    HashMapDeduplicate();
    __device__ int find(State& item, int slidesCount, State* s);
};
enum Version {
    sliding, pathfinding
};

struct Program_spec {
    Version version;
    ifstream in;
    ofstream out;
    int device;
//    Program_spec(Version version, ifstream in, ofstream out):version(version),in(in),out(out){};
};

__device__ __host__ bool operator<(const State &a, const State &b);

__device__ __host__ bool operator>(const State &a, const State &b);

__device__ __host__ void swap(State &a, State &b);

__device__ __host__ void swap(int &a, int &b);
struct PriorityQueue {
    State A[PRIORITY_QUEUE_SIZE+1];
    int lock;

    __device__ __host__ PriorityQueue();

    int heapSize = 0;

    __device__ __host__ int parent(int i);

    __device__ __host__ int left(int i);

    __device__ __host__ int right(int i);

    __device__ __host__ void maxHeapify(int i);

    __device__ __host__ void insert(State s);

    __device__ State pop();

    __device__ bool empty();

    __device__ State* top();
};

__device__ __host__ int f(const Vertex &a, const Vertex &b, int slidesCount, int slidesCountSqrt);

#endif //PROJECT1B_STRUCTURES_H

#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>
#include <regex>
#include <cmath>

namespace po = boost::program_options;
using namespace std;

const int BLOCKS_COUNT = 1;
const int THREADS_PER_BLOCK_COUNT = 1;
const int THREADS_COUNT = BLOCKS_COUNT * THREADS_PER_BLOCK_COUNT;
const int MAX_SLIDES_COUNT = 25;
const int PRIORITY_QUEUE_SIZE = 100;
const int MAX_S_SIZE = 6;
const int INF = 1000000000;
const int H_SIZE = 1024; // It must be the power of 2
int slidesCount, slidesCountSqrt;

struct Vertex {
    int slides[MAX_SLIDES_COUNT];
    __host__ __device__

    Vertex(int slides[]) {
        memcpy(this->slides, slides, MAX_SLIDES_COUNT * sizeof(int));
    }

    __device__ Vertex() {}
    __device__ __host__ int hashBase(int slidesCount, int base) {
        int result = 0;
        for(int i=0,p=1;i<slidesCount;i++,p=( p * base ) % H_SIZE) {
            result = (result + slides[i]*p) % H_SIZE;
        }
        return result;
    }
    __device__ __host__ int hash1(int slidesCount) {
        return this->hashBase(slidesCount, 30);
    }
    __device__ __host__ int hash2(int slidesCount) {
        int hash = this->hashBase(slidesCount, 29);
        hash = hash % 2 ? hash : (hash + 1) % H_SIZE;
        return hash;
    }
    __device__ __host__ int hash(int i, int slidesCount) {
        return (hash1(slidesCount) + i*hash2(slidesCount) ) % H_SIZE;
    }
    __host__ void print(int slidesCount, ostream& out) {
        for(int i=0;i<slidesCount;i++){
            if (slides[i] == 0)
                out<<"_";
            else out<<slides[i];
            if (i != slidesCount-1)
                out<<",";
        }
        out << endl;
    }
};

__device__ __host__ bool vertexEqual(const Vertex &a, const Vertex &b,const int &slidesCount) {
    for (int i = 0; i < slidesCount; i++) {
        if (a.slides[i] != b.slides[i])
            return false;
    }
    return true;
}

struct State {
    Vertex node;
    int g, f, lock;
    Vertex prev;

    __device__ __host__ State():lock(1) {}

    __device__ __host__ State(int f):f(f), lock(1) {}

    __device__ __host__ State(int g, int f, Vertex node):g(g), f(f), node(node), lock(1) {}
    __device__ __host__ State& operator=(const State& that) {
        this->node = that.node;
        this->g = that.g;
        this->f = that.f;
        this->prev = that.prev;
        return *this;
    }
};

struct HashMap {
    State hashmap[H_SIZE];
    HashMap() {
        for(int i=0;i<H_SIZE;i++)
            hashmap[i].f = -1;
    }

    __device__ __host__ State* find(Vertex& v, int slidesCount) {
        for(int i=0;i<H_SIZE;i++) {
            int hash = v.hash(i,slidesCount);
            assert(0 <= hash && hash < H_SIZE);
            if (hashmap[i].f == -1 || vertexEqual(hashmap[hash].node,v, slidesCount))
                return &hashmap[hash];
        }
        assert(false);
        return nullptr;
    }
    __device__ __host__ void insert(State& s, int slidesCount) {
        State* tmp = this->find(s.node, slidesCount);
        *tmp = s;
    }
};
enum Version {
    sliding, pathfinding
};

struct Program_spec {
    Version version;
    ifstream in;
    ofstream out;
//    Program_spec(Version version, ifstream in, ofstream out):version(version),in(in),out(out){};
};

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
struct PriorityQueue {
    State A[PRIORITY_QUEUE_SIZE];
    int lock;

    __device__ PriorityQueue():lock(1) {}

    int heapSize = 0;

    __device__ __host__ int parent(int i) {
        return i / 2;
    }

    __device__ __host__ int left(int i) {
        return i * 2;
    }

    __device__ __host__ int right(int i) {
        return i * 2 + 1;
    }

    __device__ __host__ void maxHeapify(int i) {
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

    __device__ __host__ void insert(State s) {
        assert(heapSize < PRIORITY_QUEUE_SIZE);
        heapSize++;
        A[heapSize] = s;
        int i = heapSize;
        while (i > 1 && A[parent(i)] > A[i]) {
            swap(A[i], A[parent(i)]);
            i = parent(i);
        }
    }

    __device__ State pop() {
        assert(heapSize > 0);
        State max = A[1];
        A[1] = A[heapSize];
        heapSize--;
        maxHeapify(1);
        return max;
    }

    __device__ bool empty() {
        return heapSize == 0;
    }

    __device__ State* top() {
        return (!this->empty()) ? A : nullptr;
    }
};

void parse_args(int argc, const char *argv[], Program_spec &program_spec) {
    po::options_description desc{"Options"};
    try {
        desc.add_options()
                ("version", po::value<std::string>(), "You have to specify version")
                ("input-data", po::value<std::string>())
                ("output-data", po::value<std::string>());
        po::variables_map vm;
        store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help")) {
            std::cout << desc << '\n';
            exit(0);
        }
        string version = vm["version"].as<string>();
        string input_file = vm["input-data"].as<string>();
        string output_file = vm["output-data"].as<string>();

        program_spec.in.open(input_file);
        program_spec.out.open(output_file);
        program_spec.version = version == "sliding" ? sliding : pathfinding;
    }
    catch (const po::error &ex) {
        std::cerr << ex.what() << '\n';
    }
    catch (...) {
        std::cerr << desc << '\n';
    }
}

void read_slides(ifstream &in, int *slides, int &len) {
    string s;
    getline(in, s);

    smatch m;
    regex e("_|[0-9]+");
    len = 0;
    while (regex_search(s, m, e)) {
        for (auto x:m) {
            slides[len++] = x == "_" ? 0 : stoi(x);
        }
        s = m.suffix().str();
    }
}

__device__ __host__ int f(const Vertex &a, const Vertex &b, int slidesCount, int slidesCountSqrt) {
    int pos[MAX_SLIDES_COUNT + 1];
    int sum = 0;
    for (int i = 0; i < slidesCount; i++) {
        int value = b.slides[i];
        if (value != 0) {
            assert(1 <= value && value <= slidesCount);
            pos[value] = i;
        }
    }
    for (int posA = 0; posA < slidesCount; posA++) {
        if (a.slides[posA] != 0) {
            int posB = pos[a.slides[posA]];
            int tmp1 = abs(posA % slidesCountSqrt - posB % slidesCountSqrt);
            int tmp2 = abs(posA / slidesCountSqrt - posB / slidesCountSqrt);
            sum += tmp1 + tmp2;
        }
    }
    return sum;
}

__device__ void expand(const State qi, State s[], int &sSize, const Vertex &target,int slidesCount, int
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
        State sTmp = qi; // I hope slides is copied
        sTmp.g = qi.g + 1;

        swap(sTmp.node.slides[empty], sTmp.node.slides[move]);
        sTmp.f = f(sTmp.node, target, slidesCount, slidesCountSqrt);
        sTmp.prev = qi.node;
        sTmp.lock = 1;
        assert(sSize < MAX_S_SIZE);
        s[sSize++] = sTmp;
    }
}

__host__ int calcSlidesCountSqrt(int slidesCount) {
    int slidesCountSqrt;
    for (int i = 1; i < slidesCount; i++) {
        if (i * i == slidesCount) {
            slidesCountSqrt = i;
            break;
        }
        if (i == slidesCount - 1) {
            assert(false);
        }
    }
    return slidesCountSqrt;
}

__global__ void expandKernel(Vertex *start, Vertex *target, State *m, PriorityQueue *q, State *s, int *sSize,
                             int slidesCount, int slidesCountSqrt) {

    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    sSize[id] = 0;
    if (q[id].empty()) {
        return;
    }
    State qi = q[id].pop();

    if (vertexEqual(qi.node,*target,slidesCount) ) {
        while (true) {
            int lock = atomicExch(&m->lock, 0);
            if (lock == 1) {
                if (qi.f < m->f) {
                    *m = qi;
                }
                lock = atomicExch(&m->lock, 1);
                assert(lock == 0);
                break;
            } else
                continue;
        }
    } else
        expand(qi, s + (id*MAX_S_SIZE), sSize[id], *target, slidesCount, slidesCountSqrt);
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
__global__ void removeUselessStates(HashMap *h, State *t,int *sSize, int slidesCount) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for(int i=id*MAX_S_SIZE;i < id*MAX_S_SIZE + sSize[id];i++) {
        assert(t[i].f != -1);
        State* tmp = h->find(t[i].node, slidesCount);
        if (tmp->f != -1 && tmp->g < t[i].g)
            t[i].f = -1;
    }
}
__global__ void insertNewStates(HashMap *h, State *t, int *sSize, PriorityQueue *q, int slidesCount) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for(int i=id*MAX_S_SIZE;i < id*MAX_S_SIZE + sSize[id];i++) {
        if (t[i].f != -1 ) {
            while(true) {
                State *tmp = h->find(t[i].node, slidesCount);
                int lock = atomicExch(&tmp->lock, 0);
                if (lock) {
                    if (tmp->f == -1 || tmp->g > t[i].g) {
                        *tmp = t[i];
                        int hash = tmp->node.hash1(slidesCount) % THREADS_COUNT;
                        while(true) {
                            int lock = atomicExch(&q[hash].lock, 0);
                            if (lock) {
                                q[hash].insert(t[i]);
                                int lock = atomicExch(&q[hash].lock, 1);
                                assert(lock == 0);
                                break;
                            }
                        }
                    }
                    int lock = atomicExch(&tmp->lock, 1);
                    assert(lock == 0);
                    break;
                }
            }
        }
    }
}
void printPath(HashMap &h, State &m,Vertex& start, int slidesCount, ostream& out) {
    if (vertexEqual(m.node, start, slidesCount)) {
        m.node.print(slidesCount, out);
        return;
    }
    State* tmp = h.find(m.prev,slidesCount);
    assert(tmp->f != -1);
    printPath(h,*tmp,start, slidesCount, out);
    m.node.print(slidesCount, out);
}

void main2(int argc, const char *argv[]) {
    Program_spec result;
//    parse_args(argc, argv, result);
    result.in.open("slides/2_3.in");
    result.out.open("output_data");
    result.version = sliding;
    int slides[MAX_SLIDES_COUNT], slidesCount;

    read_slides(result.in, slides, slidesCount);
    slidesCountSqrt = calcSlidesCountSqrt(slidesCount);

    Vertex start(slides);
    read_slides(result.in, slides, slidesCount);
    Vertex target(slides);

    State m(INF);
    PriorityQueue q[THREADS_COUNT];
    HashMap h;
    int sSize[THREADS_COUNT];
    State startState = State(0, f(start, target, slidesCount, slidesCountSqrt), start);
    q[0].insert(startState);
    for(int i=0;i<THREADS_COUNT;i++) {
        sSize[i] = 0;
    }
    h.insert(startState, slidesCount);

    Vertex *devStart, *devTarget;
    State *devM, *devS;
    PriorityQueue *devQ;
    HashMap *devH;
    int *devSSize, *devIsTheEnd, *devIsNotEmptyQueue;

    cudaMalloc(&devStart, sizeof(Vertex));
    cudaMalloc(&devTarget, sizeof(Vertex));
    cudaMalloc(&devM,sizeof(State));
    cudaMalloc(&devQ,sizeof(PriorityQueue) * THREADS_COUNT);
    cudaMalloc(&devS,sizeof(State) * THREADS_COUNT * MAX_S_SIZE);
    cudaMalloc(&devSSize,sizeof(int) * THREADS_COUNT);
    cudaMalloc(&devIsTheEnd, sizeof(int));
    cudaMalloc(&devIsNotEmptyQueue, sizeof(int));
    cudaMalloc(&devH, sizeof(HashMap));

    cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devTarget, &target, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devM, &m, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(devQ, q, sizeof(PriorityQueue) * THREADS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(devSSize, sSize, sizeof(int) * THREADS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(devH, &h, sizeof(HashMap), cudaMemcpyHostToDevice);


    while(true) {
        int isNotEmptyQueue = checkExistanceOfNotEmptyQueueHost(devQ,devIsNotEmptyQueue);
        if (!isNotEmptyQueue)
            break;

        expandKernel << < BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >> > (devStart, devTarget, devM, devQ, devS, devSSize,
                slidesCount, slidesCountSqrt);

        isNotEmptyQueue = checkExistanceOfNotEmptyQueueHost(devQ,devIsNotEmptyQueue);
        int isTheEnd = checkIfTheEndKernelHost(devM, devQ, devIsTheEnd);
        if (isTheEnd && isNotEmptyQueue) {
            break;
        }

        removeUselessStates <<<BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT>>>(devH, devS, devSSize, slidesCount);

        insertNewStates <<<BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT>>>(devH, devS, devSSize, devQ, slidesCount);
    }

    cudaMemcpy(&m, devM, sizeof(State), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h, devH, sizeof(HashMap), cudaMemcpyDeviceToHost);
    if (m.f == INF) {
        result.out << "path not found" << endl;
    } else {
        printPath(h, m, start, slidesCount, result.out);
    }

    cudaFree(devStart);
    cudaFree(devTarget);
    cudaFree(devM);
    cudaFree(devQ);
    cudaFree(devS);
    cudaFree(devSSize);
    cudaFree(devIsTheEnd);
    cudaFree(devH);
}

int main(int argc, const char *argv[]) {
    main2(argc, argv);
}

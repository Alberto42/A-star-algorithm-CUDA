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
const int MAX_S_SIZE = 10;
const int INF = 1000000000;
int slidesCount, slidesCountSqrt;

struct Vertex {
    int slides[MAX_SLIDES_COUNT];
    __host__ __device__

    Vertex(int slides[]) {
        memcpy(this->slides, slides, MAX_SLIDES_COUNT * sizeof(int));
    }

    __device__ Vertex() {}
};

__device__ bool vertexEqual(const Vertex &a, const Vertex &b,const int &slidesCount) {
    for (int i = 0; i < slidesCount; i++) {
        if (a.slides[i] != b.slides[i])
            return false;
    }
    return true;
}

struct State {
    Vertex node;
    int g, f, lock;
    State *prev;

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

    __device__ PriorityQueue() {}

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
};

void parse_args(int argc, const char *argv[], Program_spec &program_spec) {
    po::options_description desc{"Options"};
    try {
        desc.add_options()
                ("version", po::value<std::string>(), "You have to specify version")
                ("input_data", po::value<std::string>())
                ("output_data", po::value<std::string>());
        po::variables_map vm;
        store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help")) {
            std::cout << desc << '\n';
            exit(0);
        }
        string version = vm["version"].as<string>();
        string input_file = vm["input_data"].as<string>();
        string output_file = vm["output_data"].as<string>();

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
            slides[len++] = x == "_" ? -1 : stoi(x);
        }
        s = m.suffix().str();
    }
}

__device__ __host__ int f(const Vertex &a, const Vertex &b, int slidesCount, int slidesCountSqrt) {
    int pos[MAX_SLIDES_COUNT + 1];
    int sum = 0;
    for (int i = 0; i < slidesCount; i++) {
        int value = b.slides[i];
        if (value != -1) {
            assert(1 <= value && value <= slidesCount);
            pos[value] = i;
        }
    }
    for (int posA = 0; posA < slidesCount; posA++) {
        if (a.slides[posA] != -1) {
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
        if (slides[i] == -1) {
            empty = i;
            break;
        }
    if (empty == -1)
        assert(false);
    for (int i = 0; i < movesCount; i++) {
        int move = empty + moves[i];
        if (move < 0 || move >= slidesCount)
            continue;
        State sTmp = qi; // I hope slides is copied
        sTmp.g = qi.g + 1;

        swap(sTmp.node.slides[empty], sTmp.node.slides[move]);
        sTmp.f = f(sTmp.node, target, slidesCount, slidesCountSqrt);
        sTmp.prev = nullptr; //fixme
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

__global__ void expandKernel(Vertex *start, Vertex *target, State *m, PriorityQueue *q, State *s[], int *sSize,
                             int slidesCount, int slidesCountSqrt) {

    int id = threadIdx.x + blockIdx.x;
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
        expand(qi, s[id], sSize[id], *target, slidesCount, slidesCountSqrt);
}

void main2(int argc, const char *argv[]) {
    Program_spec result;
//    parse_args(argc, argv, result);
    result.in.open("slides/1.in");
    result.out.open("dupa");
    result.version = sliding;
    int slides[MAX_SLIDES_COUNT], slidesCount;

    read_slides(result.in, slides, slidesCount);
    slidesCountSqrt = calcSlidesCountSqrt(slidesCount);

    Vertex start(slides);
    read_slides(result.in, slides, slidesCount);
    Vertex target(slides);

    State m(INF);
    PriorityQueue q[THREADS_COUNT];
    int sSize[THREADS_COUNT];
    q[0].insert(State(0, f(start, target, slidesCount, slidesCountSqrt), start));
    for(int i=0;i<THREADS_COUNT;i++) {
        sSize[i] = 0;
    }

    Vertex *devStart, *devTarget;
    State *devM, *devS[THREADS_COUNT];
    PriorityQueue devQ[THREADS_COUNT];
    int devSSize[THREADS_COUNT];

    cudaMalloc(&devStart, sizeof(Vertex));
    cudaMalloc(&devTarget, sizeof(Vertex));
    cudaMalloc(&devM,sizeof(State));
    cudaMalloc((void**)&devQ,sizeof(PriorityQueue) * THREADS_COUNT);
    cudaMalloc((void**)&devS,sizeof(State*) * THREADS_COUNT);
    for(int i=0;i<THREADS_COUNT;i++) {
        cudaMalloc(&devS[i], sizeof(State) * MAX_S_SIZE);
    }
    cudaMalloc((void**)&devSSize,sizeof(int) * THREADS_COUNT);

    cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devTarget, &target, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devM, &m, sizeof(State), cudaMemcpyHostToDevice);
    for(int i=0;i<THREADS_COUNT;i++) {
        cudaMemcpy(devQ, q + i, sizeof(State), cudaMemcpyHostToDevice);
        cudaMemcpy(devSSize, sSize + i, sizeof(int), cudaMemcpyHostToDevice);
    }

    expandKernel << < BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >> > (devStart, devTarget, devM, devQ, devS, devSSize,
            slidesCount, slidesCountSqrt);

    cudaFree(devStart);
    cudaFree(devTarget);
    cudaFree(devM);
    cudaFree(devQ);
    cudaFree(devS);
    cudaFree(devSSize);
}

int main(int argc, const char *argv[]) {
    main2(argc, argv);
}

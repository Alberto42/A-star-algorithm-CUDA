#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>
#include <regex>
#include <cmath>
#include <time.h>
#include "kernels/expandKernel.h"
#include "structures.h"

namespace po = boost::program_options;
using namespace std;

int slidesCount, slidesCountSqrt;



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
    assert (len <= MAX_SLIDES_COUNT);
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


__global__ void improveMKernel(State *m, State *qiCandidates, int *qiCandidatesCount) {
    for(int i=0;i<*qiCandidatesCount;i++) {
        if (qiCandidates[i].f < m->f) {
            *m = qiCandidates[i];
        }
    }
    *qiCandidatesCount = 0;
}
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
        if (t[i].f == -1)
            continue;
        State* tmp = h->find(t[i].node, slidesCount);
        if (tmp->f != -1 && tmp->g < t[i].g)
            t[i].f = -1;
    }
}
__global__ void insertNewStates(HashMap *h, State *t, int *sSize, PriorityQueue *q,Vertex *target, int slidesCount,
        int slidesCountSqrt) {
    int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK_COUNT;
    for(int i=id;i < THREADS_COUNT * MAX_S_SIZE;i+=THREADS_COUNT) {
        if (t[i].f != -1) {
            t[i].f = t[i].g + f(t[i].node, *target, slidesCount,slidesCountSqrt);
            q[id].insert(t[i]);
            for(int j=0;j<H_SIZE;j++) {
                int hash = t[i].node.hash(j,slidesCount);
                assert(0 <= hash && hash < H_SIZE);
                if (h->hashmap[hash].f == -1 || vertexEqual(h->hashmap[hash].node,t[i].node, slidesCount)) {
                    int lock = atomicExch(&h->hashmap[hash].lock, 0);
                    if (lock) {
                        h->hashmap[hash] = t[i];
                        int lock = atomicExch(&h->hashmap[hash].lock, 1);
                        assert(lock == 0);
                        break;
                    }
                }
                assert(j != H_SIZE -1);
            }
        }
    }
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

void main2(int argc, const char *argv[]) {
    Program_spec result;
//    parse_args(argc, argv, result);
    result.in.open("slides/1.in");
    result.out.open("output_data");
    result.version = sliding;
    int slides[MAX_SLIDES_COUNT], slidesCount;

    read_slides(result.in, slides, slidesCount);
    slidesCountSqrt = calcSlidesCountSqrt(slidesCount);

    Vertex start(slides);
    read_slides(result.in, slides, slidesCount);
    Vertex target(slides);

    State m(INF), qiCandidates[Q_CANDIDATES_COUNT];
    PriorityQueue q;
    int sSize[THREADS_COUNT], qiCandidatesCount=0;
    State startState = State(0, f(start, target, slidesCount, slidesCountSqrt), start);
    q.insert(startState);
    for(int i=0;i<THREADS_COUNT;i++) {
        sSize[i] = 0;
    }

    Vertex *devStart, *devTarget, *devPath;
    State *devM, *devS, *devT, *devQiCandidates;
    PriorityQueue *devQ;
    HashMap *devH;
    HashMapDeduplicate *devHD;
    int *devSSize, *devIsTheEnd, *devIsNotEmptyQueue, *devQiCandidatesCount, *devPathSize;

    cudaSetDevice(1);
    cudaMalloc(&devStart, sizeof(Vertex));
    cudaMalloc(&devTarget, sizeof(Vertex));
    cudaMalloc(&devM,sizeof(State));
    cudaMalloc(&devQ,sizeof(PriorityQueue) * THREADS_COUNT);
    cudaMalloc(&devS,sizeof(State) * THREADS_COUNT * MAX_S_SIZE);
    cudaMalloc(&devT,sizeof(State) * THREADS_COUNT * MAX_S_SIZE);
    cudaMalloc(&devSSize,sizeof(int) * THREADS_COUNT);
    cudaMalloc(&devIsTheEnd, sizeof(int));
    cudaMalloc(&devIsNotEmptyQueue, sizeof(int));
    cudaMalloc(&devQiCandidatesCount, sizeof(int));
    cudaMalloc(&devQiCandidates, sizeof(State) * Q_CANDIDATES_COUNT);
    cudaMalloc(&devH, sizeof(HashMap));
    cudaMalloc(&devHD, sizeof(HashMapDeduplicate));
    cudaMalloc(&devPathSize, sizeof(int));

    cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devTarget, &target, sizeof(Vertex), cudaMemcpyHostToDevice);
    cudaMemcpy(devM, &m, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(devQ, &q, sizeof(PriorityQueue), cudaMemcpyHostToDevice);
    cudaMemcpy(devSSize, sSize, sizeof(int) * THREADS_COUNT, cudaMemcpyHostToDevice);
    cudaMemcpy(devQiCandidatesCount, &qiCandidatesCount, sizeof(int), cudaMemcpyHostToDevice);

    createHashmapKernel <<< BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >>> (devH, devStart, devTarget, slidesCount, slidesCountSqrt);

    cudaEvent_t start_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    cout<<"here1" << endl;
    while(true) {
        int isNotEmptyQueue = checkExistanceOfNotEmptyQueueHost(devQ,devIsNotEmptyQueue);
        if (!isNotEmptyQueue)
            break;

        expandKernel << < BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >> > (devStart, devTarget, devM, devQ, devS, devSSize,
                devQiCandidates, devQiCandidatesCount, slidesCount, slidesCountSqrt);
        improveMKernel <<< 1, 1 >>> (devM,devQiCandidates, devQiCandidatesCount);

        isNotEmptyQueue = checkExistanceOfNotEmptyQueueHost(devQ,devIsNotEmptyQueue);
        int isTheEnd = checkIfTheEndKernelHost(devM, devQ, devIsTheEnd);
        if (isTheEnd && isNotEmptyQueue)
            break;

        deduplicateKernelHost(devS,devSSize, devT, devHD, slidesCount);

        removeUselessStates <<<BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT>>>(devH, devT, devSSize, slidesCount);

        insertNewStates <<<BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT>>>(devH, devT, devSSize, devQ, devTarget,slidesCount,
                slidesCountSqrt);
    }

    cudaEventRecord(stop_t, 0);

    cudaMemcpy(&m, devM, sizeof(State), cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start_t, stop_t);


    result.out << elapsedTime << endl;
    if (m.f != INF) {
        cudaMalloc(&devPath, sizeof(Vertex) * (m.g+10));
        getPathKernel <<< 1, 1 >>> (devH, devM, devStart,slidesCount, devPath, devPathSize);
        int pathSize;
        cudaMemcpy(&pathSize, devPathSize, sizeof(int), cudaMemcpyDeviceToHost);
        Vertex* path = new Vertex[pathSize];
        cudaMemcpy(path, devPath, sizeof(Vertex) * pathSize, cudaMemcpyDeviceToHost);

        for(int i=pathSize-1;i>=0;i--)
            path[i].print(slidesCount,result.out);

        delete [] path;
        cudaFree(devPath);
    }


    cudaEventDestroy(start_t);
    cudaEventDestroy(stop_t);
    cudaFree(devStart);
    cudaFree(devTarget);
    cudaFree(devM);
    cudaFree(devQ);
    cudaFree(devS);
    cudaFree(devSSize);
    cudaFree(devIsTheEnd);
    cudaFree(devH);
    cudaFree(devHD);
    cudaFree(devQiCandidates);
    cudaFree(devQiCandidatesCount);
    cudaFree(devPathSize);
}

int main(int argc, const char *argv[]) {
    main2(argc, argv);
}

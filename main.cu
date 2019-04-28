#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>
#include <regex>
#include <cmath>
#include <time.h>
#include "kernels/expandKernel.h"
#include "kernels/deduplicateKernel.h"
#include "kernels/insertNewStatesKernel.h"
#include "kernels/kernels.h"
#include "kernels/removeUselessStatesKernel.h"
#include "structures.h"

namespace po = boost::program_options;
using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int slidesCount, slidesCountSqrt;

void parse_args(int argc, const char *argv[], Program_spec &program_spec) {
    po::options_description desc{"Options"};
    try {
        desc.add_options()
                ("version", po::value<std::string>(), "You have to specify version")
                ("input-data", po::value<std::string>())
                ("output-data", po::value<std::string>())
                ("device", po::value<int>()->default_value(1));
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
        int device = vm["device"].as<int>();

        program_spec.in.open(input_file);
        program_spec.out.open(output_file);
        program_spec.version = version == "sliding" ? sliding : pathfinding;
        program_spec.device = device;
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

void main2(int argc, const char *argv[]) {
    Program_spec result;
    parse_args(argc, argv, result);
    int slides[MAX_SLIDES_COUNT], slidesCount;

    read_slides(result.in, slides, slidesCount);
    slidesCountSqrt = calcSlidesCountSqrt(slidesCount);

    Vertex start(slides);
    read_slides(result.in, slides, slidesCount);
    Vertex target(slides);

    State m(INF), qiCandidates[Q_CANDIDATES_COUNT];
    PriorityQueue q;
    int sSize[THREADS_COUNT], qiCandidatesCount=0, end=0;
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
    int *devSSize, *devIsTheEnd, *devIsNotEmptyQueue, *devQiCandidatesCount, *devPathSize, *devEnd;

    gpuErrchk(cudaSetDevice(result.device));
    gpuErrchk(cudaMalloc(&devStart, sizeof(Vertex)));
    gpuErrchk(cudaMalloc(&devTarget, sizeof(Vertex)));
    gpuErrchk(cudaMalloc(&devM,sizeof(State)));
    gpuErrchk(cudaMalloc(&devQ,sizeof(PriorityQueue) * THREADS_COUNT));
    gpuErrchk(cudaMalloc(&devS,sizeof(State) * THREADS_COUNT * MAX_S_SIZE));
    gpuErrchk(cudaMalloc(&devT,sizeof(State) * THREADS_COUNT * MAX_S_SIZE));
    gpuErrchk(cudaMalloc(&devSSize,sizeof(int) * THREADS_COUNT));
    gpuErrchk(cudaMalloc(&devIsTheEnd, sizeof(int)));
    gpuErrchk(cudaMalloc(&devIsNotEmptyQueue, sizeof(int)));
    gpuErrchk(cudaMalloc(&devQiCandidatesCount, sizeof(int)));
    gpuErrchk(cudaMalloc(&devQiCandidates, sizeof(State) * Q_CANDIDATES_COUNT));
    gpuErrchk(cudaMalloc(&devH, sizeof(HashMap)));
    gpuErrchk(cudaMalloc(&devHD, sizeof(HashMapDeduplicate)));
    gpuErrchk(cudaMalloc(&devPathSize, sizeof(int)));
    gpuErrchk(cudaMalloc(&devEnd, sizeof(int)));

    gpuErrchk(cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devTarget, &target, sizeof(Vertex), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devM, &m, sizeof(State), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devQ, &q, sizeof(PriorityQueue), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devSSize, sSize, sizeof(int) * THREADS_COUNT, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devQiCandidatesCount, &qiCandidatesCount, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devEnd, &end, sizeof(int), cudaMemcpyHostToDevice));

    createHashmapKernel <<< BLOCKS_COUNT, THREADS_PER_BLOCK_COUNT >>> (devH, devStart, devTarget, slidesCount, slidesCountSqrt);

    cudaEvent_t start_t, stop_t;
    gpuErrchk(cudaEventCreate(&start_t));
    gpuErrchk(cudaEventCreate(&stop_t));
    gpuErrchk(cudaEventRecord(start_t, 0));

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
                slidesCountSqrt, devEnd);
        gpuErrchk(cudaMemcpy(&end, devEnd, sizeof(int), cudaMemcpyDeviceToHost));
        if (end == 1)
            break;
    }

    gpuErrchk(cudaEventRecord(stop_t, 0));

    gpuErrchk(cudaMemcpy(&m, devM, sizeof(State), cudaMemcpyDeviceToHost));

    float elapsedTime;
    gpuErrchk(cudaEventElapsedTime(&elapsedTime, start_t, stop_t));


    result.out << elapsedTime << endl;
    if (m.f != INF) {
        gpuErrchk(cudaMalloc(&devPath, sizeof(Vertex) * (m.g+10)));
        getPathKernel <<< 1, 1 >>> (devH, devM, devStart,slidesCount, devPath, devPathSize);
        int pathSize;
        gpuErrchk(cudaMemcpy(&pathSize, devPathSize, sizeof(int), cudaMemcpyDeviceToHost));
        Vertex* path = new Vertex[pathSize];
        gpuErrchk(cudaMemcpy(path, devPath, sizeof(Vertex) * pathSize, cudaMemcpyDeviceToHost));

        for(int i=pathSize-1;i>=0;i--)
            path[i].print(slidesCount,result.out);

        delete [] path;
        gpuErrchk(cudaFree(devPath));
    }


    gpuErrchk(cudaEventDestroy(start_t));
    gpuErrchk(cudaEventDestroy(stop_t));
    gpuErrchk(cudaFree(devStart));
    gpuErrchk(cudaFree(devTarget));
    gpuErrchk(cudaFree(devM));
    gpuErrchk(cudaFree(devQ));
    gpuErrchk(cudaFree(devS));
    gpuErrchk(cudaFree(devSSize));
    gpuErrchk(cudaFree(devIsTheEnd));
    gpuErrchk(cudaFree(devH));
    gpuErrchk(cudaFree(devHD));
    gpuErrchk(cudaFree(devQiCandidates));
    gpuErrchk(cudaFree(devQiCandidatesCount));
    gpuErrchk(cudaFree(devPathSize));
    gpuErrchk(cudaFree(devEnd));
}

int main(int argc, const char *argv[]) {
    main2(argc, argv);
}

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
//    result.in.open("slides/1.in");
//    result.out.open("output_data");
//    result.version = sliding;
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

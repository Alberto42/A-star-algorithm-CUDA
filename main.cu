#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>
#include <regex>

using namespace std;
namespace po = boost::program_options;

const int BLOCKS_COUNT = 1;
const int THREADS_PER_BLOCK_COUNT = 1;
const int THREADS_COUNT = BLOCKS_COUNT * THREADS_PER_BLOCK_COUNT
const int MAX_SLIDES_COUNT = 25;
const int PRIORITY_QUEUE_SIZE = 100;
const int MAX_S_SIZE = 100;
const int INF = 1000000000;
int slidesCount, slidesCountSqrt;
struct Vertex {
    int slides[MAX_SLIDES_COUNT];
    Vertex(int slides[]) {
        memcpy(this->slides,slides, MAX_SLIDES_COUNT * sizeof(int));
    }
    Vertex(){}
};
bool operator==(const Vertex& a, const Vertex& b) {
    for(int i=0;i<slidesCount;i++) {
        if (a.slides[i] != b.slides[i])
            return false;
    }
    return true;
}
struct State {
    Vertex node;
    int g,f;
    State *prev;
    State(){}
    State(int f):f(f){}
};
bool operator<(const State &a, const State &b) {return a.f < b.f;}
bool operator>(const State &a, const State &b) {return a.f > b.f;}

struct PriorityQueue {
    State A[PRIORITY_QUEUE_SIZE];
    PriorityQueue(){}
    int heapSize = 0;
    int parent(int i) {
        return i/2;
    }
    int left(int i) {
        return i*2;
    }
    int right(int i) {
        return i*2 + 1;
    }
    void maxHeapify(int i) {
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
            swap(A[i],A[smallest]);
            maxHeapify(smallest);
        }
    }
    void insert(State s) {
        assert(heapSize < PRIORITY_QUEUE_SIZE);
        heapSize++;
        A[heapSize] = s;
        int i=heapSize;
        while(i > 1 && A[parent(i)] > A[i]) {
            swap(A[i],A[parent(i)]);
            i = parent(i);
        }
    }
    State pop() {
        assert(heapSize > 0);
        State max = A[1];
        A[1] = A[heapSize];
        heapSize--;
        maxHeapify(1);
        return max;
    }
    bool empty() {
        return heapSize > 0;
    }
};
enum Version { sliding, pathfinding};
struct Program_spec{
    Version version;
    ifstream in;
    ofstream out;
//    Program_spec(Version version, ifstream in, ofstream out):version(version),in(in),out(out){};
};
void parse_args(int argc, const char *argv[], Program_spec& program_spec) {
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
    catch (const po::error &ex)
    {
        std::cerr << ex.what() << '\n';
    }
    catch (...) {
        std::cerr << desc << '\n';
    }
}
void read_slides(ifstream &in, int *slides, int& len) {
    string s;
    getline(in,s);

    smatch m;
    regex e ("_|[0-9]+");
    len = 0;
    while (regex_search (s,m,e)) {
        for (auto x:m) {
            slides[len++] = x == "_" ? -1 : stoi(x);
        }
        s = m.suffix().str();
    }
}
void expand(State qi, State s[],int &sSize) {
    int moves = [-1,1, -slidesCountSqrt, slidesCountSqrt];
    const int movesCount = 4;
    int empty = -1;
    int *slides = qi.node.slides;
    for(int i=0;i<slidesCount;i++)
        if (slides[i] == -1) {
            empty = i;
            break;
        }
    if (empty == -1)
        throw std::exception("empty slide not found");
    for(int i=0;i<movesCount;i++) {
        int move = empty + moves[i];
        if (move < 0 || move >= slidesCount)
            continue;
        State sTmp = qi;
        sTmp.g = qi.g + 1;
        // fixme
        swap(sTmp.node.slides[empty],sTmp.node.slides[move]);
        assert(sSize < MAX_S_SIZE);
        s[sSize++] = sTmp;
    }
}

__shared__ int m; //id in qi
__shared__ State qi[THREADS_COUNT+1];
__global__ void kernel(Vertex* start, Vertex* target) {
    PriorityQueue q;
    State s[MAX_S_SIZE];
    int sSize = 0;

    int id = threadIdx.x + blockIdx.x;
    if (id == 0) {
        q.insert(*start);
        qi[THREADS_COUNT] = State(INF);
        m = THREADS_COUNT;
    }
    __syncthreads();
    while(true) {
        sSize = 0;
        if (q.empty()) {
            continue;
        }
        qi[id] = q.pop();

        if (qi[id].node == *target) {
            int idm = id;
            while (qi[idm].f < qi[m].f) {
                idm = atomicExch(&m, idm);
            }
        }
        expand(qi[id],s,sSize);
    }

}
void calcSlidesCountSqrt() {
    for(int i=1;i<slidesCount;i++) {
        if (i*i == slidesCount) {
            slidesCountSqrt = i;
            break;
        }
        if (i == slidesCount -1) {
            throw std::exception("Wrong slides count");
        }
    }
}
void main2(int argc, const char *argv[]) {
    Program_spec result;
    parse_args(argc, argv, result);
    int slides[MAX_SLIDES_COUNT];
    read_slides(result.in, slides, slidesCount);
    calcSlidesCountSqrt();
    Vertex start(slides);

    Vertex* devStart;
    cudaMalloc(&devStart, sizeof(Vertex));

    cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice);
//    kernel<<<1,1>>>(devStart);
    cudaFree(devStart);
}

int main(int argc, const char *argv[]) {
//    main2(argc, argv);
}

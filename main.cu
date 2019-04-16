#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>
#include <regex>

using namespace std;
namespace po = boost::program_options;

const int MAX_SLIDES_COUNT = 25;
const int PRIORITY_QUEUE_SIZE = 100;
int slidesCount;
struct Vertex {
    int slides[MAX_SLIDES_COUNT];
    Vertex(int slides[]) {
        memcpy(this->slides,slides, MAX_SLIDES_COUNT * sizeof(int));
    }
    Vertex(){}
};
struct State {
    Vertex node;
    int g,f;
    State *prev;
    State(){}
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
        return i+2 + 1;
    }
    void maxHeapify(int i) {
        int l = left(i);
        int r = right(i);
        int largest;
        if (l <= heapSize && A[l] > A[i]) {
            largest = l;
        } else {
            largest = i;
        }
        if (r <= heapSize && A[r] > A[largest])
            largest = r;
        if (largest != i) {
            swap(A[i],A[largest]);
            maxHeapify(largest);
        }
    }
    void insert(State s) {
        heapSize++;
        A[heapSize] = s;
        int i=heapSize;
        while(i > 1 && A[parent(i)] < A[i]) {
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

__global__ void kernel(Vertex* start) {

}
void main2(int argc, const char *argv[]) {
    Program_spec result;
    parse_args(argc, argv, result);
    int slides[MAX_SLIDES_COUNT];
    read_slides(result.in, slides, slidesCount);
    Vertex start(slides);

    Vertex* devStart;
    cudaMalloc(&devStart, sizeof(Vertex));

    cudaMemcpy(devStart, &start, sizeof(Vertex), cudaMemcpyHostToDevice);
    kernel<<<1,1>>>(devStart);
    cudaFree(devStart);
}

int main(int argc, const char *argv[]) {
    main2(argc, argv);
}

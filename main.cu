#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

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
//                ("help", "./astar_gpu --version (sliding|pathfinding) --input-data <PATH> --output-data <PATH>")
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

int main(int argc, const char *argv[]) {
    Program_spec result;
    parse_args(argc, argv, result);
    cout << result.version << endl;
    int d;
    result.in >> d;
    result.out << d;
}

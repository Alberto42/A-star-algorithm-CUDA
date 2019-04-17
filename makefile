all: main main_gcc
main: main.cu
	nvcc -std=c++11 -lboost_program_options main.cu
main_gcc: main.cpp
	g++ -std=c++11 -lboost_program_options main.cpp -o main_gcc
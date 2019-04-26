all: main main_gcc
main: main.cu
	nvcc -std=c++11 -G -g -lboost_program_options main.cu -o astar_gpu
main_gcc: main.cpp
	g++ -std=c++11 -lboost_program_options main.cpp -o main_gcc
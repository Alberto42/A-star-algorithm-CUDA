main: main.cu
	nvcc -std=c++11 -lboost_program_options main.cu
main_gcc: main.cu
	cp main.cu main.cpp
	g++ -std=c++11 -lboost_program_options main.cpp
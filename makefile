main: main.cu
	nvcc -lboost_program_options main.cu
main_gcc: main.cu
	cp main.cu main.cpp
	g++ main.cpp
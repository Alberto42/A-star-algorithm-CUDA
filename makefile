all: astar_gpu
NVCC_FLAGS_2 = -std=c++14 -g -G
NVCC_FLAGS_R = -std=c++14 -g -G -x cu -dc
astar_gpu: main expandKernel structures
	nvcc -o astar_gpu $(NVCC_FLAGS_2) -lboost_program_options main.o structures.o kernels/expandKernel.o

main: main.cu
	nvcc -c -o main.o $(NVCC_FLAGS_R) -Xptxas -c main.cu

expandKernel: kernels/expandKernel.cu kernels/expandKernel.h structures
	nvcc -c -o kernels/expandKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/expandKernel.cu

structures: structures.cu structures.h
	nvcc -c -o structures.o $(NVCC_FLAGS_R) structures.cu
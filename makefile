all: astar_gpu
NVCC_FLAGS_2 = -std=c++14 -g -G
NVCC_FLAGS_R = -std=c++14 -g -G -x cu -dc
astar_gpu: main expandKernel structures deduplicateKernel insertNewStatesKernel kernels removeUselessStatesKernel
	nvcc -o astar_gpu $(NVCC_FLAGS_2) -lboost_program_options main.o structures.o kernels/*.o

main: main.cu
	nvcc -c -o main.o $(NVCC_FLAGS_R) -Xptxas -c main.cu

expandKernel: kernels/expandKernel.cu kernels/expandKernel.h structures
	nvcc -c -o kernels/expandKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/expandKernel.cu

deduplicateKernel: kernels/deduplicateKernel.cu kernels/deduplicateKernel.h structures
	nvcc -c -o kernels/deduplicateKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/deduplicateKernel.cu

insertNewStatesKernel: kernels/insertNewStatesKernel.cu kernels/insertNewStatesKernel.h structures
	nvcc -c -o kernels/insertNewStatesKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/insertNewStatesKernel.cu

kernels: kernels/kernels.cu kernels/kernels.h structures
	nvcc -c -o kernels/kernels.o $(NVCC_FLAGS_R) -Xptxas -c kernels/kernels.cu

removeUselessStatesKernel: kernels/removeUselessStatesKernel.cu kernels/removeUselessStatesKernel.h structures
	nvcc -c -o kernels/removeUselessStatesKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/removeUselessStatesKernel.cu

structures: structures.cu structures.h
	nvcc -c -o structures.o $(NVCC_FLAGS_R) structures.cu
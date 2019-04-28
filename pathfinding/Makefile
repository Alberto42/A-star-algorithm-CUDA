NVCC_FLAGS_2 = -std=c++14 -O3
NVCC_FLAGS_R = -std=c++14  -O3 -x cu -dc

astar_gpu: structures.o kernels/expandKernel.o kernels/deduplicateKernel.o kernels/insertNewStatesKernel.o kernels/kernels.o kernels/removeUselessStatesKernel.o main.o
	nvcc -o astar_gpu $(NVCC_FLAGS_2) -lboost_program_options main.o structures.o kernels/*.o

main.o: main.cu
	nvcc -c -o main.o $(NVCC_FLAGS_R) -Xptxas -c main.cu

kernels/expandKernel.o: kernels/expandKernel.cu kernels/expandKernel.h structures.o
	nvcc -c -o kernels/expandKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/expandKernel.cu

kernels/deduplicateKernel.o: kernels/deduplicateKernel.cu kernels/deduplicateKernel.h structures.o
	nvcc -c -o kernels/deduplicateKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/deduplicateKernel.cu

kernels/insertNewStatesKernel.o: kernels/insertNewStatesKernel.cu kernels/insertNewStatesKernel.h structures.o
	nvcc -c -o kernels/insertNewStatesKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/insertNewStatesKernel.cu

kernels/kernels.o: kernels/kernels.cu kernels/kernels.h structures.o
	nvcc -c -o kernels/kernels.o $(NVCC_FLAGS_R) -Xptxas -c kernels/kernels.cu

kernels/removeUselessStatesKernel.o: kernels/removeUselessStatesKernel.cu kernels/removeUselessStatesKernel.h structures.o
	nvcc -c -o kernels/removeUselessStatesKernel.o $(NVCC_FLAGS_R) -Xptxas -c kernels/removeUselessStatesKernel.cu

structures.o: structures.cu structures.h
	nvcc -c -o structures.o $(NVCC_FLAGS_R) structures.cu

.PHONY: clean

clean:
	rm astar_gpu *.o kernels/*.o
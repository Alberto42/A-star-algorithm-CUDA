#include <stdio.h>
#include "../cuda_utils.h"

#define NUM_TESTS 100
#define MAX_VALUE 10

int main()
{
    const int BLOCK_COUNT = 1;
    const int NUM_THREADS = BLOCK_SIZE * BLOCK_COUNT;

    int* in_host = (int*) malloc(NUM_THREADS * sizeof(int));
    int* out_host = (int*) malloc(NUM_THREADS * sizeof(int));
    int *in, *out, *temp;

    cudaMalloc(&in, NUM_THREADS * sizeof(int));
    cudaMalloc(&out, NUM_THREADS * sizeof(int));
    cudaMalloc(&temp, BLOCK_COUNT * sizeof(int));

    for (int t = 0; t < NUM_TESTS; ++t)
    {
        for (int i = 0; i < NUM_THREADS; ++i)
            in_host[i] = rand() % MAX_VALUE;

        for (int c = 0; c < BLOCK_COUNT; ++c)
        {
            int sum = 0;
            for (int i = 0; i < BLOCK_SIZE; ++i)
                sum += in_host[BLOCK_SIZE * c + i];
            /* printf("block %d : sum = %d\n", c, sum); */
        }
        cudaMemcpy(in, in_host, NUM_THREADS * sizeof(int), cudaMemcpyHostToDevice);

        prefix_scan(in, out, temp, NUM_THREADS);

        cudaMemcpy(out_host, out, NUM_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

        bool b = true;
        int sum = 0;
        for (int i = 0; i < NUM_THREADS; ++i)
        {
            if (out_host[i] != sum)
            {
                printf("[test %d] Error on position %d\n", t, i);
                b = false;
            }

            sum += in_host[i];
        }

        if (!b)
        {
            int sum = 0;
            for (int i = 0; i < NUM_THREADS; ++i)
            {
                printf("%d %d, %d\n", in_host[i], out_host[i], sum);
                sum += in_host[i];
            }
            printf("\n\n");
        }

    }

    return 0;
}

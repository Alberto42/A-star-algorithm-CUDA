#!/usr/bin/env bash


blockCounts=(1 1 3 1 10 100)
threadPerBlockCounts=(1 3 1 32 32 32)
for ((i=1;i<6;i+=1))
do
    cp main.cu main_tmp.cu
    sed -i -E "s/(const int BLOCKS_COUNT = )[0-9]+;/\1${blockCounts[$i]};/" main_tmp.cu
    sed -i -E "s/(const int THREADS_PER_BLOCK_COUNT = )[0-9]+;/\1${threadPerBlockCounts[$i]};/" main_tmp.cu
    echo "compile:"
    cat < main_tmp.cu | grep "const int BLOCKS_COUNT"
    cat < main_tmp.cu | grep "const int THREADS_PER_BLOCK_COUNT"
    nvcc -std=c++11 -G -g -lboost_program_options main_tmp.cu -o out_script
    echo "run:"
    ./out_script
    rm main_tmp.cu
    echo "end"

done

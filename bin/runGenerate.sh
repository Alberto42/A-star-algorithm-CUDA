#!/usr/bin/env bash


blockCounts=(1 1 3 3)
threadPerBlockCounts=(1 3 1 3)
#for ((i=0;i<4;i+=1))
#do
#    cp main.cu main_tmp.cu
#    sed -i -E "s/(const int BLOCKS_COUNT = )[0-9]+;/\1${blockCounts[$i]};/" main_tmp.cu
#    sed -i -E "s/(const int THREADS_PER_BLOCK_COUNT = )[0-9]+;/\1${threadPerBlockCounts[$i]};/" main_tmp.cu
#    echo "compile:"
#    cat < main_tmp.cu | grep "const int BLOCKS_COUNT"
#    cat < main_tmp.cu | grep "const int THREADS_PER_BLOCK_COUNT"
#    nvcc -std=c++11 -G -g -lboost_program_options main_tmp.cu -o out_script_${blockCounts[$i]}_${threadPerBlockCounts[$i]}
#
#done

for((j=0;j<10000;j+=1))
do
    python3 generate.py 3 $j 10 > generated
    echo "**********generated $j"
    cat < generated
    echo "***********************"
    for ((i=0;i<4;i+=1))
    do
        echo "**********output $j $i"
        ./out_script_${blockCounts[$i]}_${threadPerBlockCounts[$i]} --version sliding --input-data generated \
        --output-data output_data_${blockCounts[$i]}_${threadPerBlockCounts[$i]}
        DIFF=$(diff output_data_${blockCounts[$i]}_${threadPerBlockCounts[$i]} output_data_1_1)
        if [ "$DIFF" != "" ]
        then
            diff output_data_${blockCounts[$i]}_${threadPerBlockCounts[$i]} output_data_1_1
            exit 1
        fi

        cat < output_data_${blockCounts[$i]}_${threadPerBlockCounts[$i]}
        echo "***********************"
    done

done
ls

#for ((i=0;i<4;i+=1))
#do
#    rm out_script_${blockCounts[$i]}_${threadPerBlockCounts[$i]}
#done
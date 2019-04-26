#!/bin/bash

NUM_TESTS_PASSED=0

for i in {1..100}
do
    echo "Running test ${i}..."
    INPUT_DATA_PATH="tests/sliding_test_${i}.in"
    OUTPUT_DATA_PATH="tests/sliding_test_${i}.out"
    OUTPUT_DATA_TEST_PATH="tests/sliding_test_${i}.test_out"

    python3 sliding_generate_mine.py 3 $i 100 > ${INPUT_DATA_PATH}
#    python3 sliding_generate.py > ${INPUT_DATA_PATH}
    ../astar_gpu --version sliding --input-data ${INPUT_DATA_PATH} --output-data ${OUTPUT_DATA_PATH}
    ./sliding_test ${INPUT_DATA_PATH} ${OUTPUT_DATA_TEST_PATH}
    origin_time=$(head -n 1 ${OUTPUT_DATA_PATH})
    echo "Times = ${origin_time}"
    python3 sliding_check.py ${INPUT_DATA_PATH} ${OUTPUT_DATA_PATH} ${OUTPUT_DATA_TEST_PATH}
    val=$?
    if [ $val -ne 0 ]
    then
        echo -e "\e[31mTest ${i} failed.\e[0m"
    else
        echo -e "\e[32mTest ${i} succeeded.\e[0m"
        NUM_TESTS_PASSED=$((NUM_TESTS_PASSED + 1))
    fi
done

echo "Tests passed: ${NUM_TESTS_PASSED}/100"

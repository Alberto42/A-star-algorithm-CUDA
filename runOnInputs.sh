#!/usr/bin/env bash

for filename in slides/*.in; do
    echo "filename: ***************** $filename"
    ./runOnDifferentNumberOfThreads.sh "$filename"
    echo "filename end: ***************** $filename"

done
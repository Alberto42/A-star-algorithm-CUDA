#!/usr/bin/env bash

./astar_gpu --version slides --input-data $1 --output-data output_data
cat < output_data
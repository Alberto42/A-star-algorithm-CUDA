#!/usr/bin/env bash

./astar_gpu --version slides --input-data $1 --output-data output_data --device 3
cat < output_data
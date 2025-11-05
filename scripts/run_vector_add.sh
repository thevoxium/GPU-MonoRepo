#!/bin/bash

numbers=(256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

for n in "${numbers[@]}"; do
    echo "Running for input: $n"
    ../out "$n"
    echo "--------------------"
done

#!/bin/bash
cp ${1} kernel.cu
if [[ 0 == $(make) ]] ; do
    echo "Make Success!"
done
# for j in {1..2} ; do
#     for i in 1000 10000 100000 ; do
#         ./mandelbrot -g 1 -v ${j} -i ${i}
#     done
# done    
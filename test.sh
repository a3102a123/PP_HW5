#!/bin/bash
cp ${1} kernel.cu
rm mandelbrot
make
if [ `ls | grep mandelbrot$` ] ; then
    for j in {1..2} ; do
        for i in 1000 10000 100000 ; do
            ./mandelbrot -g 1 -v ${j} -i ${i}
        done
    done
fi
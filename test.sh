#!/bin/bash
if [ $(ls | grep ^mandelbrot$) ] ; then
    #echo "remove man"
    rm mandelbrot
fi
if [ $(ls | grep ^kernel.cu$) ] ; then
    #echo "remove kernel"
    rm kernel.cu
fi
if [ ${1} ] ; then
    #echo "cp kernel" 
    cp ${1} kernel.cu
else
    echo "Usage : bash test.sh kernel_file"
    exit -1
fi
make
if [ $(ls | grep ^mandelbrot$) ] ; then
    #echo "True"
    for j in {1..2} ; do
        for i in 1000 10000 100000 ; do
            echo "Running ${1} -v ${j} -i ${i}"
            ./mandelbrot -g 1 -v ${j} -i ${i}
        done
    done
fi
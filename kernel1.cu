#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define XBLOCK_SIZE 32
#define YBLOCK_SIZE 24

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY,int width,int count, int *output) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float c_re = lowerX + i * stepX;
    float c_im = lowerY + j * stepY;

    int idx;
    float z_re = c_re, z_im = c_im;
    for (idx = 0; idx < count; ++idx)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
        break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    int index = (j * width + i);
    output[index] = idx;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int size = resX * resY * sizeof(int);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    // allocate memory in host & device
    int *host_mem, *dev_mem;
    host_mem = (int *)malloc(size);
    cudaMalloc((void **)&dev_mem, size);
    // GPU processing 
    dim3 num_block(resX / XBLOCK_SIZE, resY / YBLOCK_SIZE);
    dim3 block_size(XBLOCK_SIZE, YBLOCK_SIZE);
    mandelKernel<<<num_block, block_size>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, dev_mem);
    cudaDeviceSynchronize();
    // GPU translate result data back
    cudaMemcpy(host_mem, dev_mem, size, cudaMemcpyDeviceToHost);
    memcpy(img, host_mem, size);
    free(host_mem);
    cudaFree(dev_mem);
}

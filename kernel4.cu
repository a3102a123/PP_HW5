#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define XBLOCK_SIZE 16
#define YBLOCK_SIZE 48

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY,int width,int count, int *output, int round, int round_size) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + round * round_size;
    int index = (j * width + i);
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

    output[index] = idx;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int size = resX * resY * sizeof(int),j;
    int round_size = resX * YBLOCK_SIZE * sizeof(int);
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    // allocate memory in host & device
    int *host_mem1, *host_mem2, *host_ptr, *host_pre_ptr, *dev_mem,*temp_ptr;
    int n = resY / YBLOCK_SIZE;
    cudaStream_t stream[n],lock; 
    cudaStreamCreate ( &lock );
    for(j = 0 ; j < n ; j++ )
        cudaStreamCreate ( &stream[j] );
    cudaHostAlloc(&host_mem1, round_size, cudaHostAllocDefault);
    cudaHostAlloc(&host_mem2, round_size, cudaHostAllocDefault);
    cudaMalloc((void **)&dev_mem, size);
    host_ptr = host_mem1;
    host_pre_ptr = host_mem1;
    // GPU processing 
    dim3 num_block(resX / XBLOCK_SIZE, 1);
    dim3 block_size(XBLOCK_SIZE, YBLOCK_SIZE);
    for(j = 0 ; j < n ; j++){
        mandelKernel<<<num_block, block_size,0 , stream[j]>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, dev_mem, j, YBLOCK_SIZE);
    }
    for(j = 0 ; j < n ; j++ ){
        cudaStreamSynchronize( stream[j] );
        cudaMemcpyAsync( host_ptr, dev_mem + (round_size / sizeof(int)) * j, round_size, cudaMemcpyDeviceToHost,lock);
        if(j >= 0 ){
            cudaStreamSynchronize( lock );
            memcpy(img + (round_size / sizeof(int)) * (j), host_pre_ptr, round_size);
        }
        temp_ptr = host_ptr;
        host_ptr = host_pre_ptr;
        host_pre_ptr = temp_ptr;
    }
    cudaStreamSynchronize( lock );
    memcpy(img + (round_size / sizeof(int)) * (j - 1), host_pre_ptr, round_size);
    /*for(int j = 25 ; j < 28 ; j++){
        for(int i = 0 ; i < resX ; i++)
            printf("%d ",host_mem[i + j * resX]);
        printf("\n");
    }*/
    // GPU translate result data back
    cudaFreeHost(host_mem1);
    cudaFreeHost(host_mem2);
    cudaFree(dev_mem);
}

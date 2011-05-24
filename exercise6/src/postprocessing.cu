#include <glog/logging.h>
#include <iostream>
#include "postprocessing.h"

#if __GPUVERSION__

#define chunksizex 20
#define chunksizey 16

texture<rgb, 2, cudaReadModeElementType> textureImage;

__host__ __device__ rgb mix(const rgb& r1, const rgb& r2, const rgb& r3, const rgb& r4)
{
    rgb result;
    result.x = (r1.x + r2.x + r3.x + r4.x) >> 2;
    result.y = (r1.y + r2.y + r3.y + r4.y) >> 2;
    result.z = (r1.z + r2.z + r3.z + r4.z) >> 2;
    return result;
}

__global__ void antialiasepixel(rgb* result, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < (width - 1) && y < (height - 1))
    {
        result[y * (width - 1) + x] = mix(tex2D(textureImage, x, y),
                                          tex2D(textureImage, x + 1, y),
                                          tex2D(textureImage, x, y + 1),
                                          tex2D(textureImage, x + 1, y + 1));
    }
}
#endif

void antialiase(int height, int width, rgb* image)
{
#if __GPUVERSION__
    //create and bind texture from source image
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

    cudaError_t error;
    rgb* d_image;
    size_t pitch;

    error = cudaMallocPitch(&d_image, &pitch, width * sizeof(rgb), height);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaMemcpy2D(d_image, pitch, image, width * sizeof(rgb), width * sizeof(rgb) , height, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);


    error = cudaBindTexture2D(0, &textureImage, d_image, &channelDesc, width, height, pitch);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    //Malloc memory for result
    rgb *d_result;
    error = cudaMalloc(&d_result, (width - 1) * (height - 1) * sizeof(rgb) );
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    dim3 threadsPerBlock(chunksizex, chunksizey);
    dim3 blocksPerGrid((width + chunksizex - 2) / chunksizex, (height + chunksizey - 2) / chunksizey);

    //execute kernel
    antialiasepixel <<< blocksPerGrid, threadsPerBlock>>>(d_result, width, height);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    //copy back result
    error = cudaMemcpy(image, d_result, (width - 1) * (height - 1) * sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    cudaUnbindTexture(textureImage);
    cudaFree(d_image);
    cudaFree(d_result);
#endif
}
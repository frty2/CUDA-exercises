#include <stdio.h>

__global__ void emptyKernel()
{
    printf("empty kernel call\n");
}

int main()
{
    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);
    emptyKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaThreadSynchronize();
    return 0;
}

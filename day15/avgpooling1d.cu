#include <cuda_runtime.h>
#include <iostream>
using namespace std; 

__global__ void solutionkernel(const float* input, int kernel_size, int stride, int padding, float* output, size_t H,int outh)
{
    int outi=blockIdx.x*blockDim.x+threadIdx.x;
    float pvalue=0.0f;

    int instart=outi*stride-padding;
    if(outi<outh)
    {



    for(int i=0;i<kernel_size;i++)
    {

        if(instart+i>=0 &&instart+i<H)
        {
            pvalue+=input[instart+i];
        }
    }


    output[outi]=pvalue/kernel_size;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {  


    int outh=((H+(2*padding)-kernel_size)/stride)+1;
    size_t blockSize=256;
    size_t numblocks=(float(outh)+blockSize-1)/blockSize;
    cout<<"the num of blocks is: "<<numblocks;

    solutionkernel<<<numblocks,blockSize>>>(input,kernel_size,stride,padding,output,H, outh);


      
}
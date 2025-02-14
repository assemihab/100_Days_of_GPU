#include <iostream>
#include <fstream>

#define tilesize 2
using namespace std;


__global__ void binary_prodcut(int width)
{
    __shared__ bool md[tilesize][tilesize];
    int phases=width/tilesize;
    int by=blockIdx.y,ty=threadIdx.y;
    int bx=blockIdx.x,tx=threadIdx.x;
    
    for (int p=0;p<phases;p++)
    {
        int loadmx=p*tilesize+tx;
        int loadmy=by*tilesize+ty;
        int loadny=p*tilesize+ty;
        int loadnx=bx*tilesize+tx;

        printf("Phase %d\n",p);
        printf("the entry loaded by thread T[%d][%d] is M[%d][%d] and N[%d][%d]\n",by*tilesize+ty,bx*tilesize+tx,loadmy,loadmx,loadny,loadnx);
        // __syncthreads();
    }
     
}

int main()
{   
    ifstream ifile("matrix1.txt");
    int rows,cols;
    ifile>>rows>>cols;
    bool*matrix1=new bool[rows*cols];
    for (int i=0;i<rows*cols;i++)
    {
        ifile>>matrix1[i];
    }
    for (int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
            cout<<matrix1[(i*cols)+j];
        }
        cout<<endl;
    }
    bool*matd,*matdout;
    size_t sizee=rows*cols*sizeof(bool);
    cudaMalloc(&matd,sizee);
    cudaMalloc(&matdout,sizee);

    cudaMemcpy(matd,matrix1,sizee,cudaMemcpyHostToDevice);
    
    dim3 dimGrid(2,2);
    dim3 dimBlock(2,2);
    binary_prodcut<<<dimGrid,dimBlock>>>(tilesize*2);
    cudaDeviceSynchronize();
    cudaFree(matd);
    cudaFree(matdout);
    delete[] matrix1;


    return 0;
}
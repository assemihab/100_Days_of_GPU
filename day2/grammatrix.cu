#include <iostream>
#include<fstream>

using namespace std;

__global__ void grammatrix(int *md, int *outd, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int multaddition=0;
    

    if (row < rows && col < cols)
    {
        for (int i=0;i<cols;i++)
        {
            multaddition+=md[row*cols+i]*md[col*cols+i];
        }
        outd[row*cols+col]=multaddition;
        printf("md[%d][%d]=%d\n",row,col,outd[row*cols+col]);
    }
}
int main() 
{
    // declaring variables
    int rows,cols;
    int **matrix1;
    ifstream ifile("mat.txt");
    if (!ifile)
    {
        cout<<"error opening file!"<<endl;
        return 0;
    }
    ifile>>rows>>cols;

    matrix1=new int*[rows];
    for (int i=0;i<rows;i++)
    {
        matrix1[i]=new int[cols];
    }
    int *rowflatmat;
    rowflatmat=new int [rows*cols];
    for (int i=0;i<rows;i++)
    {
        for(int j=0;j<cols;j++)
        {
           ifile>> matrix1[i][j];
           rowflatmat[i*cols+j]=matrix1[i][j];
        }
    }
    int sizee= rows*cols*sizeof(int);
    int *md;
    int *outd;
    cudaMalloc(&md,sizee);
    cudaMalloc(&outd,sizee);
    cudaMemcpy(md,rowflatmat,sizee,cudaMemcpyHostToDevice);
    float blocksize=32;

    dim3 dimblock(blocksize,blocksize);
    dim3 dimgrid((unsigned int)ceil(rows/blocksize),(unsigned int)ceil(cols/blocksize));
    // cout << ceil(rows/blocksize)<<endl;
    grammatrix<<<dimgrid,dimblock>>>(md,outd,rows,cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

}
#include<iostream>
#include<fstream>

using namespace std;

__global__ void binaryproduct(bool *A, bool*C, int rows, int cols)
{
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    for (int col=0;col<cols;col++)
    {
        int sum=0;
    if(row>=rows || col>=cols)
        return;
    for(int i=0; i<cols; i++)
        sum+= A[row*cols+i] * A[i*cols+col];
    if (sum>0)
        C[row*cols+col] = 1;
    else
        C[row*cols+col] = 0;
    }
    
}

int main()
{
    ifstream ifile("binmat.txt");
    int rows, cols;
    ifile >> rows >> cols;
    bool *A = new bool[rows*cols];
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            ifile >> A[i*cols+j];
    ifile.close();

    bool *dmat;
    cudaMalloc(&dmat, rows*cols*sizeof(bool));
    cudaMemcpy(dmat, A, rows*cols*sizeof(bool), cudaMemcpyHostToDevice);

    bool *dres;
    cudaMalloc(&dres, rows*rows*sizeof(bool));

    dim3 block(3,1);
    dim3 grid(1,1);
    binaryproduct<<<grid, block>>>(dmat, dres, rows, cols);

    bool *res = new bool[rows*cols];
    cudaMemcpy(res, dres, rows*cols*sizeof(bool), cudaMemcpyDeviceToHost);

    ofstream ofile("binmatprod.txt");
    ofile << rows << " " << cols << endl;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
            ofile << res[i*cols+j] << " ";
        ofile << endl;
    }
    ofile.close();

    delete[] A;
    delete[] res;
    cudaFree(dmat);
    cudaFree(dres);
    return 0;

}
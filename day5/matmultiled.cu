#include <iostream>
#include <fstream> 
using namespace std;

# define tilesize 2

__global__ void matmultiled(short* mat1, short* mat2, short* mat3, int width) {
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    short sum = 0;
    __shared__ short s1[tilesize][tilesize], s2[tilesize][tilesize];
    int phases=(width+tilesize-1)/tilesize;
    for (int p = 0; p < phases; p++) {
        if (by * tilesize + ty < width && p * tilesize + tx < width) {
            s1[ty][tx] = mat1[(by * tilesize + ty) * width + p * tilesize + tx];
        }
        else {
            s1[ty][tx] = 0;
        }
        if (p * tilesize + ty < width && bx * tilesize + tx < width) {
            s2[ty][tx] = mat2[(p * tilesize + ty) * width + bx * tilesize + tx];
        }
        else {
            s2[ty][tx] = 0;
        }
        __syncthreads();
        for (int i = 0; i < tilesize; i++) {
            sum += s1[ty][i] * s2[i][tx];
        }
        __syncthreads();
    }
    if (by * tilesize + ty < width && bx * tilesize + tx < width) {
        mat3[(by * tilesize + ty) * width + bx * tilesize + tx] =(short) sum;
    }
    

}

int main() {
    short *mat1, *mat2;
    ifstream file1("mat1.txt");
    ifstream file2("mat2.txt");
    short rows1, cols1, rows2, cols2;
    file1 >> rows1 >> cols1;
    file2 >> rows2 >> cols2;


    mat1 = new short[rows1 * cols1];
    mat2 = new short[rows2 * cols2];
    for (int i = 0; i < rows1 * cols1; i++) {
        file1 >> mat1[i];
    }
    for (int i = 0; i < rows2 * cols2; i++) {
        file2 >> mat2[i];
    }
    short*mat1d, *mat2d, *mat3d;
    cudaMalloc(&mat1d, rows1 * cols1 * sizeof(short));
    cudaMalloc(&mat2d, rows2 * cols2 * sizeof(short));
    cudaMalloc(&mat3d, rows1 * cols2 * sizeof(short));
    cudaMemcpy(mat1d, mat1, rows1 * cols1 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(mat2d, mat2, rows2 * cols2 * sizeof(short), cudaMemcpyHostToDevice);
    dim3 dimGrid((cols2 + tilesize - 1) / tilesize, (rows1 + tilesize - 1) / tilesize);
    dim3 dimBlock(tilesize, tilesize);
    matmultiled << <dimGrid, dimBlock >> > (mat1d, mat2d, mat3d, rows1);
    short* mat3 = new short[rows1 * cols2];
    cudaMemcpy(mat3, mat3d, rows1 * cols2 * sizeof(short), cudaMemcpyDeviceToHost);


    ofstream file3("mat3.txt");
    file3 << rows1 << " " << cols2 << endl;
    for (int i = 0; i < rows1 * cols2; i++) {
        file3 << mat3[i] << " ";
    }

    delete[] mat1;
    delete[] mat2;
    delete[] mat3;
    cudaFree(mat1d);
    cudaFree(mat2d);
    cudaFree(mat3d);
    return 0;
}
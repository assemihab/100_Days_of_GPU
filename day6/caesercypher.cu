#include <iostream>
#include <fstream>

using namespace std;

__global__ void encrypt(char *input, char *output, int key) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = input[idx] + key;
}

__global__ void decrypt(char *input, char *output, int key) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    output[idx] = input[idx] - key;
}


int main() 
{
    ifstream file("unencrypted.txt");
    string str;
    getline(file, str);
    file.close();
    char *din, *dout;
    char *input = new char[str.length()];
    char *output = new char[str.length()];
    strcpy(input, str.c_str());
    cudaMalloc(&din, str.length());
    cudaMalloc(&dout, str.length());

    cudaMemcpy(din, input, str.length(), cudaMemcpyHostToDevice);
    int blocksize=10;
    dim3 dimGrid(ceil(str.length()+blocksize-1/blocksize), 1, 1);
    dim3 dimBlock(blocksize, 1, 1);

    encrypt<<<dimGrid, dimBlock>>>(din, dout, 3);
    cudaMemcpy(output, dout, str.length(), cudaMemcpyDeviceToHost);
    cout << "Encrypted: " << output << endl;
    
    char*decrypted;
    cudaMalloc(&decrypted, str.length());
    decrypt<<<dimGrid, dimBlock>>>(dout, decrypted, 3);
    cudaMemcpy(output, decrypted, str.length(), cudaMemcpyDeviceToHost);
    cout << "Decrypted: " << output << endl;
    return 0;

    }
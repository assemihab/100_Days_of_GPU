#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Define block size for CUDA threads

__global__ void conv1d(const float *input, const float *kernel, float *output, int input_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size - kernel_size + 1) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            sum += input[idx + j] * kernel[j];
        }
        output[idx] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int input_size = 10;
    const int kernel_size = 3;
    const int output_size = input_size - kernel_size + 1;

    float h_input[input_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_kernel[kernel_size] = {1, 0, -1};
    float h_output[output_size];

    float *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size * sizeof(float)), "Allocating input");
    checkCudaError(cudaMalloc(&d_kernel, kernel_size * sizeof(float)), "Allocating kernel");
    checkCudaError(cudaMalloc(&d_output, output_size * sizeof(float)), "Allocating output");

    checkCudaError(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice), "Copying input");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice), "Copying kernel");

    int grid_size = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    conv1d<<<grid_size, BLOCK_SIZE>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost), "Copying output");

    std::cout << "Output: ";
    for (int i = 0; i < output_size; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

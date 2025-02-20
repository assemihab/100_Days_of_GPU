#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Define block size for CUDA threads

__global__ void conv2d(const float *input, const float *kernel, float *output, int input_width, int input_height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_k = kernel_size / 2;
    if (x >= half_k && y >= half_k && x < input_width - half_k && y < input_height - half_k) {
        float sum = 0.0f;
        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                sum += input[(y + i) * input_width + (x + j)] * kernel[(i + half_k) * kernel_size + (j + half_k)];
            }
        }
        output[y * input_width + x] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int input_width = 5, input_height = 5;
    const int kernel_size = 3;
    float h_input[input_width * input_height] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
    float h_kernel[kernel_size * kernel_size] = { 1, 0, -1, 1, 0, -1, 1, 0, -1 };
    float h_output[input_width * input_height];

    float *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_width * input_height * sizeof(float)), "Allocating input");
    checkCudaError(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)), "Allocating kernel");
    checkCudaError(cudaMalloc(&d_output, input_width * input_height * sizeof(float)), "Allocating output");

    checkCudaError(cudaMemcpy(d_input, h_input, input_width * input_height * sizeof(float), cudaMemcpyHostToDevice), "Copying input");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice), "Copying kernel");

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((input_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (input_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, input_width, input_height, kernel_size);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(h_output, d_output, input_width * input_height * sizeof(float), cudaMemcpyDeviceToHost), "Copying output");

    std::cout << "Output:\n";
    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            std::cout << h_output[y * input_width + x] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8  // Define block size for CUDA threads

__global__ void conv3d(const float *input, const float *kernel, float *output, int width, int height, int depth, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int half_k = kernel_size / 2;
    if (x >= half_k && y >= half_k && z >= half_k && x < width - half_k && y < height - half_k && z < depth - half_k) {
        float sum = 0.0f;
        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                for (int k = -half_k; k <= half_k; k++) {
                    sum += input[(z + k) * width * height + (y + j) * width + (x + i)] * 
                           kernel[(k + half_k) * kernel_size * kernel_size + (j + half_k) * kernel_size + (i + half_k)];
                }
            }
        }
        output[z * width * height + y * width + x] = sum;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int width = 4, height = 4, depth = 4;
    const int kernel_size = 3;
    float h_input[width * height * depth];
    float h_kernel[kernel_size * kernel_size * kernel_size];
    float h_output[width * height * depth];

    for (int i = 0; i < width * height * depth; i++) {
        h_input[i] = 1.0f;  // Example initialization
    }
    for (int i = 0; i < kernel_size * kernel_size * kernel_size; i++) {
        h_kernel[i] = 1.0f;  // Example kernel
    }

    float *d_input, *d_kernel, *d_output;
    checkCudaError(cudaMalloc(&d_input, width * height * depth * sizeof(float)), "Allocating input");
    checkCudaError(cudaMalloc(&d_kernel, kernel_size * kernel_size * kernel_size * sizeof(float)), "Allocating kernel");
    checkCudaError(cudaMalloc(&d_output, width * height * depth * sizeof(float)), "Allocating output");

    checkCudaError(cudaMemcpy(d_input, h_input, width * height * depth * sizeof(float), cudaMemcpyHostToDevice), "Copying input");
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice), "Copying kernel");

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE, (depth + BLOCK_SIZE - 1) / BLOCK_SIZE);
    conv3d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, depth, kernel_size);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(h_output, d_output, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost), "Copying output");

    std::cout << "Output:\n";
    for (int z = 0; z < depth; z++) {
        std::cout << "Slice " << z << ":\n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                std::cout << h_output[z * width * height + y * width + x] << " ";
            }
            std::cout << std::endl;
        }
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

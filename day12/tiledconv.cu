#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16 // Tile size (adjust based on hardware constraints)
#define KERNEL_SIZE 3 // Kernel size (assuming square)

__global__ void conv2D_tiled(const float* __restrict__ input, 
                             const float* __restrict__ kernel, 
                             float* output, 
                             int width, int height) {
    // Shared memory tile
    __shared__ float tile[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    int halo = KERNEL_SIZE / 2;

    // Load tile with boundary check
    if (row < height && col < width) {
        tile[ty][tx] = input[row * width + col];
    } else {
        tile[ty][tx] = 0.0f; // Padding for out-of-bounds values
    }
    __syncthreads();

    // Compute convolution only for valid output pixels
    if (tx < TILE_SIZE && ty < TILE_SIZE && row >= halo && row < height - halo && col >= halo && col < width - halo) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                sum += tile[ty + i][tx + j] * kernel[i * KERNEL_SIZE + j];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    // Image and kernel size
    int width = 1024, height = 1024;
    int imgSize = width * height * sizeof(float);
    int kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    // Allocate host memory
    float *h_input = new float[width * height];
    float *h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];
    float *h_output = new float[width * height];

    // Initialize input and kernel
    for (int i = 0; i < width * height; i++) h_input[i] = 1.0f;
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) h_kernel[i] = 1.0f;

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_kernel, kernelSize);
    cudaMalloc(&d_output, imgSize);

    // Copy data to device
    cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    conv2D_tiled<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy output back to host
    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    std::cout << "Convolution completed successfully!" << std::endl;
    return 0;
}

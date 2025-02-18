#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

__global__ void blurimage(unsigned char *input, unsigned char* output, int cols, int rows, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        for (int c = 0; c < channels; c++)  // Process each color channel separately
        {
            int index = (y * cols + x) * channels + c;
            int sum = 0;
            int count = 0;

            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int x1 = x + i;
                    int y1 = y + j;
                    
                    if (x1 >= 0 && x1 < cols && y1 >= 0 && y1 < rows)
                    {
                        int index1 = (y1 * cols + x1) * channels + c;
                        sum += input[index1];
                        count++;
                    }
                }
            }
            output[index] = sum / count; 
        }
    }
}

int main()
{
    Mat image=imread("cat.jpeg");
        // Allocate memory for the output image
    int rows = image.rows;
    int cols = image.cols;
    
    Mat imageblured(rows, cols, CV_8UC3); // Store the blurred image
    unsigned char *dinput, *doutput;
    int size=image.rows*image.cols*image.channels();
    cudaMalloc(&dinput,size*sizeof(unsigned char));
    cudaMalloc(&doutput,size*sizeof(unsigned char));
    cudaMemcpy(dinput,image.data,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
    dim3 block(32,32);
    dim3 grid((image.cols+block.x-1)/block.x,(image.rows+block.y-1)/block.y);
    blurimage<<<grid,block>>>(dinput,doutput,image.cols,image.rows,image.channels());
    cudaMemcpy(imageblured.data,doutput,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);
    cudaFree(dinput);
    cudaFree(doutput);
    imshow("Blured Image",imageblured);
    waitKey(0);
}
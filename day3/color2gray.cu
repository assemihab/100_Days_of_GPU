#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


__global__ void colored2gray(unsigned char* dinput,unsigned char*doutput,int height,int width,int channels)
{
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    

    if (y<height && x<width)
    {
        int redIdx=((y*width+x)*channels)+2;
    int blueIdx=((y*width+x)*channels)+0;
    int greendIdx=((y*width+x)*channels)+1;
    int currentIdx=y*width+x;
        float redeq=0.299*dinput[redIdx];
        float blueeq=0.587*dinput[blueIdx];
        float greeneq=0.114*dinput[greendIdx];
        doutput[currentIdx]=(unsigned char)redeq+blueeq+greeneq;
    }
}

int main()
{
    Mat image=imread("cat.jpeg");
    if (image.empty())
    {
        cout<<"error occured"<<endl;
        return 0;
    }
    int width=image.cols;
    int height=image.rows;

    int channels=image.channels();
    Mat gray_image(height, width, CV_8UC1, Scalar(0));

    int imgSize=width*height*channels;
    // for (int i=0;i<imgSize;i++)
    // {
    //     cout<<"\n"<<image.data[i];
    // }
    unsigned char *dinput,*doutput;
    size_t inputsize=width*height*channels*sizeof(unsigned char);
    size_t outputsize=width*height*sizeof(unsigned char);
    cudaMalloc(&dinput,inputsize);
    cudaMalloc(&doutput,outputsize);

    cudaMemcpy(dinput,image.data,inputsize,cudaMemcpyHostToDevice);
    float threads=32;
    float yblocks=ceil(height/threads);
    float xblocks=ceil(width/threads);
    cout<<endl<<"the number of blocks of y is: "<<yblocks;
    cout<<endl<<"the number of blocks of x is: "<<xblocks;
    cout<<endl<<"what's happening here";

    dim3 dimBlock(32,32);
    dim3 dimGrid(xblocks,yblocks);
    colored2gray<<<dimGrid,dimBlock>>>(dinput,doutput,(int)height,(int)width,(int)channels);
    cudaMemcpy(gray_image.data,doutput,outputsize,cudaMemcpyDeviceToHost);
    cudaFree(doutput);
    cudaFree(dinput);
    imshow("Original Image", image);
    imshow("Grayscale Image", gray_image);
    waitKey(0);

}
#include <iostream>

using namespace std;


__global__ void addingTwovecs(double *Da,double*Db,double*Dc, int n)
{
    int ithelement=blockIdx.x*blockDim.x+threadIdx.x;
    if(ithelement<n)
    {

        Dc[ithelement]=Da[ithelement]+Db[ithelement];
    }
}


int main()
{
    int n=5;
    double *arr1;
    double *arr2;
    double *arr3;
    arr3=new double[n];
    arr1=new double[n];
    arr2=new double[n];

    for (int i=0;i<n;i++)
    {
        arr1[i]=i+0.5;
        arr2[i]=i+1.5;
    }
    


    for (int i=0;i<n;i++)
    {
        
        cout<<"\n"<<"\n"<<*(arr1+i);
        cout<<"\nthe ith element of the summed array: "<<arr3[i];
    }
    
    double *ad,*bd,*cd;
    const int sizee=n*sizeof(double);

    cudaMalloc(&ad,sizee);
    cudaMalloc(&bd,sizee);
    cudaMalloc(&cd,sizee);
    float threads=32;
    float blocks=ceil(n/threads);
    cout<<"the number of blocks are: "<<blocks;
    cudaMemcpy(ad,arr1,sizee,cudaMemcpyHostToDevice);
    cudaMemcpy(bd,arr2,sizee,cudaMemcpyHostToDevice);
    
    addingTwovecs<<<blocks,threads>>>(ad,bd,cd,n);

    cudaMemcpy(arr3,cd,sizee,cudaMemcpyDeviceToHost);
    for (int i=0;i<n;i++)    for (int i=0;i<n;i++)
    {
        cout<<"\n"<<arr1[i];
        cout<<"\nthe ith element of the summed array: "<<arr3[i];
    }




}
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

#define MAX_ARRAY_SIZE 2048
#define RANDOM_MAX  2.0
#define RANDOM_MIN  1.0
#define TILE_WIDTH 32
#define EPSILON 0.000001
#define NUM_BLOCKS (MAX_ARRAY_SIZE/TILE_WIDTH)

float A[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
float F[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
float C[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];

void serial();
void init_F();
int check();
__global__ void matrixMultiply1(float *, float *, int);
__global__ void matrixMultiply2(float *, float *, int);
__global__ void matrixMultiply3(float *, float *, int);


int main()
{
	float *d_a, *d_c;
	struct timeval startTime, endTime;
	size_t memsize = MAX_ARRAY_SIZE * MAX_ARRAY_SIZE * sizeof(float);

	cudaMalloc((void**) &d_a, memsize);
	cudaMalloc((void**) &d_c, memsize);

	init_F();

	cudaMemcpy(d_a,A,memsize,cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,C,memsize,cudaMemcpyHostToDevice);

	gettimeofday(&startTime, NULL);
    
    //serial();
	//dim3 dimGrid1(1,1);
	//dim3 dimBlock1(MAX_ARRAY_SIZE, MAX_ARRAY_SIZE); 

	dim3 dimGrid2(MAX_ARRAY_SIZE/TILE_WIDTH, MAX_ARRAY_SIZE/TILE_WIDTH);
	dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH);
	matrixMultiply1<<< dimGrid2, dimBlock2 >>>(d_a,d_c,MAX_ARRAY_SIZE);

	//matrixMultiply2<<< dimGrid2, dimBlock2 >>>(d_a,d_c,MAX_ARRAY_SIZE);*/
	//matrixMultiply3<<< dimGrid2, dimBlock2 >>>(d_a,d_c,MAX_ARRAY_SIZE);
	
	gettimeofday(&endTime, NULL);

	long seconds  = endTime.tv_sec  - startTime.tv_sec;
   	long useconds = endTime.tv_usec - startTime.tv_usec;

    double duration = seconds + useconds/1000000.0;
    cout<<"\nTime taken for Matrix Multiplication on GPU (time in sec): "<<fixed<<setprecision(7)<<duration;
    cout<<"\nPerformance Metrics (GFlops/sec):"<<fixed<<setprecision(6)<<((2 * (long)MAX_ARRAY_SIZE * MAX_ARRAY_SIZE * MAX_ARRAY_SIZE))/(1e9 * duration);
    cout<<endl;

	cudaMemcpy(C,d_c,memsize,cudaMemcpyDeviceToHost);

    if(check() == 1) {
    	cout<<"\nMatrix Multiplication Successful!"<<endl;
    }

	cudaFree(d_a);
	cudaFree(d_c);

	return 0;
}

void init_F()
{
	srand(time(NULL));

	for (int i = 0; i < MAX_ARRAY_SIZE; i++){
		for (int j = 0; j < MAX_ARRAY_SIZE; j++){
			float r = ((float)rand()) / (float)RAND_MAX;
			A[i][j] = RANDOM_MIN + r * (RANDOM_MAX - RANDOM_MIN);
		}
	}
}

__global__ void matrixMultiply1(float *A, float *C, int size) {
	int Col = blockDim.y * blockIdx.y + threadIdx.y;
	int Row = blockDim.x * blockIdx.x + threadIdx.x;


	for(int k = 0; k < size; k++)
		C[Row * size + Col] += A[k * size + Row] * A[k * size + Col];

}

__global__ void matrixMultiply2(float* A, float* C, int size)
{
	float sum = 0;
 	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
 	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
 	
 	if(Col < size && Row < size) {
  		for (int k = 0; k < size; k++)
   			sum += A[k * size + Row] * A[k * size + Col];

  		C[Row * size + Col] = sum;
 	}
}

__global__ void matrixMultiply3(float* A, float* C, int size) {
    	
    float CValue = 0;
        
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    		 
    for (int k = 0; k < (TILE_WIDTH + size - 1)/TILE_WIDTH; k++) {
    			
        if (k * TILE_WIDTH + threadIdx.x < size && Row < size)	
          	As[threadIdx.y][threadIdx.x] = A[Row * size + k * TILE_WIDTH + threadIdx.x];
    	else													
    		As[threadIdx.y][threadIdx.x] = 0.0;

    	if (k * TILE_WIDTH + threadIdx.y < size && Col < size)	
    		As[threadIdx.y][threadIdx.x] = A[(k*TILE_WIDTH + threadIdx.y) * size + Col];
    	else													
    		As[threadIdx.y][threadIdx.x] = 0.0;
             
    	__syncthreads();

    	for (int n = 0; n < TILE_WIDTH; ++n) 
    		CValue += As[threadIdx.y][n] * As[n][threadIdx.x];
    		
    	__syncthreads();
        }
        
    if (Row < size && Col < size) 
       	C[((blockIdx.y * blockDim.y + threadIdx.y) * size) + (blockIdx.x*blockDim.x) + threadIdx.x] = CValue;
}

void serial()
{
	for (int i = 0; i < MAX_ARRAY_SIZE; i++)
		for (int j = 0; j < MAX_ARRAY_SIZE; j++) 
			for (int k = 0; k < MAX_ARRAY_SIZE; k++)
				F[i][j] += A[k][i] * A[k][j]; 
}

int check()
{
	for (int i = 0; i < MAX_ARRAY_SIZE; i++) {
		for (int j = 0; j < MAX_ARRAY_SIZE; j++) {
			if(abs(C[i][j] - F[i][j]) < EPSILON){
				cout<<"\nMismatch at index: ("<<i<<","<<j<<")"<<endl;
				return 0;
			}
		}
	}
	return 1;
}
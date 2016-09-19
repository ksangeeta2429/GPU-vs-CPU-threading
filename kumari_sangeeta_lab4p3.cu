#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

#define MAX_ARRAY_SIZE 1024
#define RANDOM_MAX  1000
#define TILE_DIM 16
#define BLOCK_ROWS 8
#define EPSILON 0.000001
#define NUM_BLOCKS (MAX_ARRAY_SIZE/TILE_DIM)

float A[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
float C[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];

void serial();
void init_F();
int check();
__global__ void matrixTranspose1(float *);
__global__ void matrixTranspose2(const float *, float *);

int main()
{
	float *d_a;
	//float *d_c;
	struct timeval startTime, endTime;
	size_t memsize = MAX_ARRAY_SIZE * MAX_ARRAY_SIZE * sizeof(float);

	cudaMalloc((void**) &d_a, memsize);
	//cudaMalloc((void**) &d_c, memsize);

	init_F();

	cudaMemcpy(d_a,A,memsize,cudaMemcpyHostToDevice);
	//cudaMemcpy(d_c,C,memsize,cudaMemcpyHostToDevice);

	gettimeofday(&startTime, NULL);

    //serial(); 
    
 	dim3 dimGrid2(MAX_ARRAY_SIZE/TILE_DIM, MAX_ARRAY_SIZE/TILE_DIM);
	dim3 dimBlock2(TILE_DIM, TILE_DIM);

	matrixTranspose1<<< 16, 1024 >>>(d_a);
	//matrixTranspose2<<< dimGrid2, dimBlock2 >>>(d_a,d_c);
	
	gettimeofday(&endTime, NULL);

	double seconds  = endTime.tv_sec  - startTime.tv_sec;
    double useconds = endTime.tv_usec - startTime.tv_usec;

    double duration = seconds + useconds/1000000.0;
    cout<<"\nTime taken for Matrix Transpose on GPU (time): "<<fixed<<setprecision(7)<<duration<<endl;
	cudaMemcpy(C,d_a,memsize,cudaMemcpyDeviceToHost);

  	if(check() == 1) {
    	cout<<"\nMatrix Transpose Successful!"<<endl;
    }

	cudaFree(d_a);
	return 0;
}

void init_F()
{
	srand(time(NULL));  

	for (int i = 0; i < MAX_ARRAY_SIZE; i++) {
		for (int j = 0; j < MAX_ARRAY_SIZE; j++) {
			A[i][j] = rand() % RANDOM_MAX;
		}
	}
}

__global__ void matrixTranspose1(float *A) {
	int width = MAX_ARRAY_SIZE / gridDim.x;
	
	for(int i = blockIdx.x * width; i < blockIdx.x * width + width; i++) {
		int rowWidth = i / blockDim.x + 1;

		for(int j = threadIdx.x * rowWidth; j < i && j < threadIdx.x * rowWidth + rowWidth; j++) {
			float temp = A[i * MAX_ARRAY_SIZE + j];
			A[i * MAX_ARRAY_SIZE + j] = A[j * MAX_ARRAY_SIZE + i];
			A[j * MAX_ARRAY_SIZE + i] = temp;
		}
	}
}

__global__ void matrixTranspose2(const float *F, float *C)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
    
  	int x = blockIdx.x * TILE_DIM + threadIdx.x;
  	int y = blockIdx.y * TILE_DIM + threadIdx.y;
  	int width = gridDim.x * TILE_DIM;

  	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    	tile[threadIdx.y+j][threadIdx.x] = F[(y+j)*width + x];

  	__syncthreads();

  	x = blockIdx.y * TILE_DIM + threadIdx.x; 
  	y = blockIdx.x * TILE_DIM + threadIdx.y;

  	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    	C[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


void serial()
{
	for (int i = 0; i < MAX_ARRAY_SIZE; i++) {
		for (int j = (i+1); j < MAX_ARRAY_SIZE; j++) {
			float temp = A[i][j];
			A[i][j] = A[j][i];
			A[j][i] = temp;
		}
	}
}

int check()
{
	for (int i = 0; i < MAX_ARRAY_SIZE; i++) {
		for (int j = 0; j < MAX_ARRAY_SIZE; j++) {
			if(abs(C[i * MAX_ARRAY_SIZE + j] - A[j * MAX_ARRAY_SIZE + i]) < EPSILON){
				cout<<"\nMismatch at index: ("<<i<<","<<j<<")"<<endl;
				return 0;
			}
		}
	}
	return 1;
}
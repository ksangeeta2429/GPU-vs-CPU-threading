#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <sys/time.h>
#include <cuda.h>

using namespace std;

#define MAX_ARRAY_SIZE 4096
#define RANDOM_MAX  2.0
#define RANDOM_MIN  1.0

float A[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];
float F[MAX_ARRAY_SIZE][MAX_ARRAY_SIZE];

void serial();
void init_F();
void print();
__global__ void matrixFunc(float *, int);

int main()
{
	float *d_a;
	struct timeval startTime, endTime;
	size_t memsize = MAX_ARRAY_SIZE * MAX_ARRAY_SIZE * sizeof(float);

	cudaMalloc((void**) &d_a, memsize);

	init_F();

	cudaMemcpy(d_a,F,memsize,cudaMemcpyHostToDevice);

	gettimeofday(&startTime, NULL);
    
    //serial();

	matrixFunc<<< 1,1 >>>(d_a, MAX_ARRAY_SIZE);
	
	gettimeofday(&endTime, NULL);

	double seconds  = endTime.tv_sec  - startTime.tv_sec;
    double useconds = endTime.tv_usec - startTime.tv_usec;

    double duration = seconds + useconds/1000000.0;
    cout<<"\nTime taken for computation on GPU (time in sec): "<<fixed<<setprecision(7)<<duration;
    cout<<"\nPerformance Metrics (GFlops/sec):"<<fixed<<setprecision(6)<< (100 * (long)MAX_ARRAY_SIZE * MAX_ARRAY_SIZE)/ (1e9 * duration);
    cout<<endl;

	cudaMemcpy(F,d_a,memsize,cudaMemcpyDeviceToHost);

	cudaFree(d_a);

	return 0;
}

void init_F()
{
	srand(time(NULL));

	for (int i = 0; i < MAX_ARRAY_SIZE; i++){
		for (int j = 0; j < MAX_ARRAY_SIZE; j++){
			float r = ((float)rand()) / (float)RAND_MAX;
			A[i][j] = F[i][j] = RANDOM_MIN + r * (RANDOM_MAX - RANDOM_MIN);
		}
	}
}

__global__ void matrixFunc(float *F, int size)
{
	#pragma unroll 16
	for(int k = 0; k < 100; k++)
		#pragma unroll 16
		for(int i = 1; i < size; i++)
			for(int j = 0; j < size - 1; j++)
				F[i * size + j] = F[(i-1) * size + j + 1] + F[i * size + j + 1];
}

void serial()
{
	for (int k = 0; k < 100; k++)
		for (int i = 1; i <= MAX_ARRAY_SIZE; i++)
			for (int j = 0; j < MAX_ARRAY_SIZE; j++)
				A[i][j] = A[i - 1][j + 1] + A[i][j + 1];
}


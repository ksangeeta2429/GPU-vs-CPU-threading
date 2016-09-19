// Matrix Multiply

#include<stdio.h>

// perform MatrixMul on Device
__global__ void MatrixMulDevice( float *A, float *B, float *C, int *matrixSize)
{
    int chunk = (*matrixSize) / gridDim.x;
    int sum, i, k;

    for(i = blockIdx.x * chunk; i < blockIdx.x * chunk + chunk - 1; i++) {
        sum = 0;

        for(k = 0; k < *matrixSize; k++) {
            sum += A[i * *matrixSize + k] * B [k * *matrixSize + threadIdx.x];
        }

        C[i * *matrixSize + threadIdx.x] = sum;
    }
}

int MatrixMulHostValidate(float *A, float *B, float *C, int dim)
{

float a, b, sum;
for (int i= 0; i< dim; i++)
{
for (int j = 0; j < dim; j++)
{
sum = 0;
for (int k = 0; k < dim; k++) {
a = A[ i* dim + k ];
b = B[ k * dim + j ];
sum += a * b;
}

 if (C[ i* dim + j ] != sum)
    return 0;
}
}

return 1;
}

void initMatrix(float *A, int dim) {
for (int i= 0; i< dim; i++)
{
    for (int j = 0; j < dim; j++)
    {
        A[i* dim + j] = ((float)i + j) / dim;
    }
}
}

int main(void) {
float *A, *B, *C;
int dim = 512;
float *d_A, *d_B, *d_C;
int *d_matrixSize;

// Allocate memory for the matrices.
A = (float *) malloc(sizeof(float) * dim * dim);
B = (float *) malloc(sizeof(float) * dim * dim);
C = (float *) malloc(sizeof(float) * dim * dim);

// I/O to load A, B and C.
initMatrix(A, dim);
initMatrix(B, dim);

// define thread hierarchy
int nblocks= 4;
int tpb= 512;

// allocate device memory
size_t memSize;
memSize= dim * dim * sizeof(float);
cudaMalloc( (void**) &d_A, memSize);
cudaMalloc( (void**) &d_B, memSize);
cudaMalloc( (void**) &d_C, memSize);
cudaMalloc( (void**) &d_matrixSize, sizeof(float));

// initialize device memory
cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, memSize, cudaMemcpyHostToDevice);
cudaMemcpy(d_matrixSize, &dim, sizeof(float), cudaMemcpyHostToDevice);

// launch kernel
dim3 dimGrid(nblocks);
dim3 dimBlock(tpb);
// perform MatrixMulon Device
MatrixMulDevice<<< dimGrid, dimBlock>>>(d_A, d_B, d_C, d_matrixSize);

// retrieve results
cudaMemcpy(C, d_C, memSize, cudaMemcpyDeviceToHost);

// verfiy results
if(!MatrixMulHostValidate(A, B, C, dim))
    fprintf(stderr, "Wrong results for matrix multiply\n");
else
    printf("Matrix multiply was successful\n");

// Free memory
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
cudaFree(d_matrixSize);
free(A);
free(B);
free(C);
}

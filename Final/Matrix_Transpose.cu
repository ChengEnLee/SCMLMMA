#include<stdio.h>
#include<stdlib.h>

int M,N;
double *A, *AT;
double *d_A, *d_AT;

__global__ void MT(double *A, double *AT, int m, int n){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx < m){
		for(int rows=0; rows<n; rows++){
			AT[idx * n + rows] = A[idx + rows * m];
		}
	}
}

int main(int argc, char *argv[]){

	M = atoi(argv[1]);
	N = atoi(argv[2]);

	A = (double *)malloc(M*N*sizeof(double));
	AT = (double *)malloc(M*N*sizeof(double));

	for(int i=0;i<M*N;i++) A[i] = i;
	for(int i=0;i<M*N;i++) AT[i] = 0;

	cudaMalloc((void**) &d_A, M*N*sizeof(double));
	cudaMalloc((void**) &d_AT, M*N*sizeof(double));

	cudaMemcpy(d_A, A, M*N*sizeof(double), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	dim3 block(256);
	dim3 grid((255+M)/256);

	cudaEventRecord(start);
	for(int i=0;i<5;i++) MT<<<grid,block>>>(d_A,d_AT,M,N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // this is the runtime for 100 spmvs in ms
    printf("runtime [ms]: %f\n", milliseconds/ 5.0 );

	cudaMemcpy(AT, d_AT, M*N*sizeof(double), cudaMemcpyDeviceToHost);

	free(A);
	free(AT);
	cudaFree(d_A);
	cudaFree(d_AT);

}

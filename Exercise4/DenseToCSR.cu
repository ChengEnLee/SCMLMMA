#include <stdio.h>
#include <stdlib.h>

int M,N; //M rows, N cols
int *h_row, *h_col, *d_row, *d_col, nnz;
double *h_A, *h_value, *d_A, *d_value;

//CPU Compute
void compute_nnz_row(double *A, int *row){

	for(int rows=0; rows<M; rows++){
		for(int cols=0; cols<N; cols++){
			if(A[rows * N + cols] != 0){
				row[rows+1] += 1;
			}
		}
	}
}

__global__ void compute_nnz_row_gpu(double *A, int *row, int M, int N){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < M){
		for(int cols = 0; cols<N; cols++){
			if(A[idx * N + cols] != 0) row[idx+1] += 1;
		}
	}
}

int compute_nnz(int *row){
	for(int idx = 0; idx < M; idx++){
		row[idx+1] += row[idx];
	}
	return  row[M];
}

__global__ void compute_nnz_gpu(int *row, int M){
	
	//int idx = blockDim.x * blockIdx.x + threadIdx.x;
	for(int idx = 0; idx<M; idx++){
		row[idx+1] += row[idx];
		__syncthreads();
	}
}

void compute_nnz_col(double *A, int *col, double *value){

	int id = 0;
	for(int rows = 0; rows < M; rows++){
		for(int cols = 0; cols < N; cols++){
			if(A[rows * N + cols] != 0){
				value[id] = A[rows * N + cols];
				col[id] = cols;
				id += 1;
			}
		}
	}
}

__global__ void compute_nnz_col_gpu(double *A, int *col, double *value, int M, int N){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int id = 0;
	if(idx < M){
		for(int cols = 0; cols < N; cols++){
			if(A[idx * N + cols] != 0){
				value[id] = A[idx * N + cols];
				col[id] = cols;
				id += 1;
			}
		}
	}
}

int main(int argc, char *argv[]){

	M = atoi(argv[1]);
	N = atoi(argv[1]);
	// M = 30000;
	// N = 30000;

	h_A = (double *)malloc( M*N*sizeof(double));
	h_row = (int *)malloc( (M+1)*sizeof(int));
	cudaMalloc((void**) &d_A, M*N*sizeof(double));
	cudaMalloc((void**) &d_row, (M+1)*sizeof(int));

	//Initial Matrix
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			if( (i+j)%2 == 0 ){
				h_A[ i * N + j] = 1;
			}
			else{
				h_A[ i * N + j] = 0;
			}
		}
	}
	for(int i=0; i<M+1; i++) h_row[i] = 0;

	cudaMemcpy(d_A, h_A, M*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, h_row, (M+1)*sizeof(int), cudaMemcpyHostToDevice);

/*	double start, end;
	start = clock();
	//First, we compute nonzero element in every row.
	compute_nnz_row(h_A, h_row);
	
	//Report the total nonzero element by sum of each row.
	nnz = compute_nnz(h_row);
	free(h_row);
	//After we know the nnz, we can allocate the nnz_value, nnz_col.
	h_value = (double *)malloc( nnz * sizeof(double));
	h_col = (int *)malloc( nnz * sizeof(int));

	//Compute every nonzero element column index.
	compute_nnz_col(h_A, h_col, h_value);
	end = clock(); */


	//////////////
	// GPU Compute Start
	cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);
	compute_nnz_row_gpu<<< (M+255)/256, 256>>>(d_A, d_row, M, N);
	// compute_nnz_gpu<<<(M+255)/256, 256>>>(d_row, M);
	cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);


////////////
	cudaMemcpy(h_row, d_row, (M+1)*sizeof(int), cudaMemcpyDeviceToHost);

	double start3, end3;
	start3 = clock();
	compute_nnz(h_row);
	end3 = clock();

	nnz = h_row[M];

	cudaMalloc((void**) &d_value, nnz*sizeof(double));
	cudaMalloc((void**) &d_col, nnz*sizeof(int));

	cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
	compute_nnz_col_gpu<<< (nnz+255)/256, 256>>>(d_A, d_col, d_value, M, N);
	cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    cudaMemcpy(h_value, d_value, nnz*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col, d_col, nnz*sizeof(int), cudaMemcpyDeviceToHost);

    float milliseconds1 = 0;
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("runtime : %f (s)\n", (milliseconds1 + milliseconds2)*1e-3 + (end3 - start3)/CLOCKS_PER_SEC);
    //GPU Part End

	//Print all of nonzero index
	printf("we have %i nonzero values. \n",nnz);
	//printf("Dense to CSR elapsed time = %f\n", (end - start)/CLOCKS_PER_SEC);
	/*
	for(int vecs = 0; vecs < M; vecs++){
		for(int i = h_row[vecs];i<h_row[vecs+1];i++){
			printf("(%i,%i) = %f\n", vecs, h_col[i], h_value[i]);
		}
	}
	*/

	//Free all memory we use.
	cudaFree(d_A);
	cudaFree(d_value);
	cudaFree(d_row);
	cudaFree(d_col);
	free(h_A);
	free(h_value);
	free(h_row);
	free(h_col);

}
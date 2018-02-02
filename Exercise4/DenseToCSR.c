#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int M,N; //M rows, N cols
int *h_row, *h_col, nnz;
double *h_A, *h_value;

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


int compute_nnz(int *row){
	for(int idx = 0; idx < M; idx++){
		row[idx+1] += row[idx];
	}
	return  row[M];
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

int main(int argc, char *argv[]){

	M = atoi(argv[1]);
	N = atoi(argv[1]);
	//printf("%i %i\n",M,N);

	h_A = (double *)malloc( M*N*sizeof(double));
	h_row = (int *)malloc( (M+1)*sizeof(int));

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


	double start, end;
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
	end = clock(); 

	//Print all of nonzero index
	printf("we have %i nonzero values. \n",nnz);
	printf("Dense to CSR elapsed time = %f\n", (end - start)/CLOCKS_PER_SEC);
	/*
	for(int vecs = 0; vecs < M; vecs++){
		for(int i = h_row[vecs];i<h_row[vecs+1];i++){
			printf("(%i,%i) = %f\n", vecs, h_col[i], h_value[i]);
		}
	}
	*/

	//Free all memory we use.
	 free(h_A);
	 free(h_value);
	 free(h_col);
}

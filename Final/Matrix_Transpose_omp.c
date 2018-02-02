#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define max(x,y) (x > y ? x : y)

int M,N;
int *A, *A_T;
//int *temp;
int tnum;
//Matrix Storage Type : Col Major :

// void M_transpose(int *A){

// 	#pragma omp parallel for num_threads(tnum)
// 	for(int cols=0; cols<M; cols++){
// 		int temp = 0;
// 		for(int rows=cols+1; rows<N; rows++){
// 			temp = A[cols * N + rows];
// 			A[cols * N + rows] = A[cols + rows * M];
// 			A[cols + rows * M] = temp;
// 			//A_T[cols * N + rows] = A[cols + rows * M];
// 		}
// 	}
// }

void M_transpose(int *A, int *A_T){

	#pragma omp parallel for num_threads(tnum)
	for(int cols=0; cols<M; cols++){
		for(int rows=0; rows<N; rows++){
			A_T[cols * N + rows] = A[cols + rows * M];
		}
	}
}

int main(int argc, char *argv[]){

	//Matrix MxN (row, col)
	M = atoi(argv[1]);
	N = atoi(argv[2]);
	tnum = 8;
	//int vec = max(M,N);
	//Allocate Memory
	size_t size;
	size = M*N*sizeof(int);
	A = (int *)malloc(size);
	A_T = (int *)malloc(size);

	//Init
	for(int i=0; i<M*N; i++) A[i] = i;
	for(int i=0; i<M*N; i++) A_T[i] = 0;

	//Transpose
	double start, end;
	start = omp_get_wtime();
	for(int t=0;t<10;t++) M_transpose(A,A_T);
	end = omp_get_wtime();
	printf("elapsed time = %f s\n", (end - start)/10.0);//CLOCKS_PER_SEC /tnum);
	//Print(A, A_T);

	free(A);
	free(A_T);
}

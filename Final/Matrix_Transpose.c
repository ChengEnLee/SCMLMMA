#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int M,N;
int *A, *A_T;

//Matrix Storage Type : Col Major :

void M_transpose(int *A, int *A_T){

	for(int cols=0; cols<M; cols++){
		for(int rows=0; rows<N; rows++){
			A_T[cols * N + rows] = A[cols + rows * M];
		}
	}
}

// void M_transpose(int *A){

// 	int temp = 0;
// 	for(int cols=0; cols<N; cols++){
// 		for(int rows=cols+1; rows<N; rows++){
// 			temp = A[cols * N + rows];
// 			A[cols * N + rows] = A[cols + rows * M];
// 			A[cols + rows * M] = temp;
// 		}
// 	}
// }


void Print(int *A){

	printf("A = \n");
	for(int rows=0; rows<M; rows++){
		for(int cols=0; cols<N; cols++){
			printf("%i ",A[rows + cols * M]);
		}
		printf("\n");
	}
	printf("\n");

	// printf("A_T = \n");
	// for(int rows=0; rows<N; rows++){
	// 	for(int cols=0; cols<M; cols++){
	// 		printf("%f ",A_T[rows + cols * N]);
	// 	}
	// 	printf("\n");
	// }
}

int main(int argc, char *argv[]){

	//Matrix MxN (row, col)
	M = atoi(argv[1]);
	N = atoi(argv[2]);
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
	start = clock();
	// M_transpose(0,2,A);
	M_transpose(A,A_T);
	end = clock();
	
	printf("elapsed time = %f s\n", (end - start)/CLOCKS_PER_SEC);
	//Print(A, A_T);

	free(A);
	free(A_T);

}

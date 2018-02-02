#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <cmath>
#include <algorithm>

using namespace std;

int main(int argc, char const *argv[]){
	clock_t start_time, end_time;
	srand((unsigned)time(NULL));

	int N;
	cout << "Please enter size of A: ";
	cin >> N;

	int *C;
	double *L, *U, *Q;
	L = (double*)malloc(N*N*sizeof(double));
	U = (double*)malloc(N*N*sizeof(double));
	Q = (double*)malloc(N*N*sizeof(double));
	C = (int*)malloc(N*sizeof(int));

  	// Ax = b
	cout << "Please enter A: " << endl;
	double* A;
	A = (double*)malloc(N*N*sizeof(double));

	// Initialize A, L and Q as identity, U as A
	for(int i = 0; i < N; i++) for(int j = 0; j < N; ++j) cin >> A[i*N+j];
	for(int i = 0; i < N; i++) for(int j = 0; j < N; ++j) U[i*N+j] = A[i*N+j];
	for(int i = 0; i < N; i++) for(int j = 0; j < N; ++j) if(i == j) L[i*N+j] = 1; else L[i*N+j] = 0;
	for(int i = 0; i < N; i++) for(int j = 0; j < N; ++j) Q[i*N+j] = 0;
	for(int i = 0; i < N; i++) C[i] = i;

	cout << "matrix A is: " << endl;
 	for(int i = 0; i < N; ++i){
    	for(int j = 0; j < N; ++j) cout << A[i*N+j] << '\t';
      	cout << endl;
  	}

  	// Updating L and U
  	for(int k = 0; k < N; k++){
  		// Find max
  		int maxidx = k;
  		for(int i = k+1; i < N; i++) if(abs(U[k*N+i]) > abs(U[k*N+maxidx])) maxidx = i;
  		swap(C[k], C[maxidx]);

  		// Exchange two columns
  		double *tmpcol;
  		tmpcol = (double*)malloc(N*sizeof(double));
  		for(int i = 0; i < N; i++) tmpcol[i] = U[i*N+k];
  		for(int i = 0; i < N; i++) U[i*N+k] = U[i*N+maxidx];
  		for(int i = 0; i < N; i++) U[i*N+maxidx] = tmpcol[i];
  		free(tmpcol);

	    // row: i, col: j
	    for(int i = k+1; i < N; i++){
	      	double scaling = U[i*N+k] / U[k*N+k];
	      	L[i*N+k] = scaling;
	      	for(int j = k; j < N; j++) U[i*N+j] -= U[k*N+j] * scaling;
	    }
  	}

  	// Update Q
  	for(int i = 0; i < N; i++) Q[i*N+C[i]] = 1;


  	cout << "matrix L is: " << endl;
 	for(int i = 0; i < N; i++){
    	for(int j = 0; j < N; ++j) cout << L[i*N+j] << '\t';
      	cout << endl;
  	}

  	cout << "matrix U is: " << endl;
 	for(int i = 0; i < N; i++){
    	for(int j = 0; j < N; ++j) cout << U[i*N+j] << '\t';
      	cout << endl;
  	}

  	cout << "matrix Q is: " << endl;
 	for(int i = 0; i < N; i++){
    	for(int j = 0; j < N; ++j) cout << Q[i*N+j] << '\t';
      	cout << endl;
  	}

	free(L);
	free(U);
	free(A);
	free(Q);
	free(C);

	return 0;
}

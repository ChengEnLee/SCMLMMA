#ifndef DGEMM_TEST_H_
#define DGEMM_TEST_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double *A, *B, *C, *x, *y;
int m;
int alpha, beta;

void Init_V(){

  //printf ("\n This example computes real matrix C=alpha*A*B+beta*C using.. \n\n");
  printf("\n Initial x,y\n");

  printf("m = \t");
  scanf("%d", &m);
/*
  printf("k = \t");
  scanf("%d", &k);
  printf("n = \t");
  scanf("%d", &n);
*/
//  m = 128*(order), k = 128*(order), n = 128*(order);
/*
  printf("Initializing data for dgemm C=A*B... \n"
	 "A(%ix%i) and B(%ix%i)... \n\n", m, k, k, n);
*/
//  alpha = 1, beta = 0;
//  printf("alpha = %i, beta = %i \n\n", alpha, beta);

  printf("Allocating... \n\n");

  //A = (double *)malloc( m*m*sizeof( double ));
  //B = (double *)malloc( m*m*sizeof( double ));
  x = (double *)malloc( m*sizeof( double ));
  y = (double *)malloc( m*sizeof( double ));

  //C = (double *)malloc( m*n*sizeof( double ));

  printf("Initializing vector data \n\n");
/*
  for(int i = 0; i < (m*m); i++){
      A[i] = 0;
  }

  for(int i = 0; i < (m*m); i++){
      B[i] = 0;
  }
*/
  for(int i = 0;i < m; i++){
      x[i] = 0;
      y[i] = 0;
  }


  printf("Initializing Success \n\n");
}

void Init_M(){

  //printf ("\n This example computes real matrix C=alpha*A*B+beta*C using.. \n$
  printf("\n Initial A,B\n");

  printf("m = \t");
  scanf("%d", &m);

  printf("Allocating... \n\n");

  A = (double *)malloc( m*m*sizeof( double ));
  B = (double *)malloc( m*m*sizeof( double ));
  C = (double *)malloc( m*m*sizeof( double ));
  //x = (double *)malloc( m*sizeof( double ));
  //y = (double *)malloc( m*sizeof( double ));

  //C = (double *)malloc( m*n*sizeof( double ));

  printf("Initializing matrix data \n\n");

  for(int i = 0; i < (m*m); i++){
      A[i] = 0;
  }

  for(int i = 0; i < (m*m); i++){
      B[i] = 0;
  }

  for(int i = 0; i < (m*m); i++){
      C[i] = 0;
  }

  printf("Initializing Success \n\n");
}


void Free_V(){

  //free(A);
  //free(B);
  free(x);
  free(y);

}

void Free_M(){

  free(A);
  free(B);
  free(C);
}

/*
double norm(double *x){

  double temp = 0;
  for(int i=0;i<m;i++){
    temp += x[i]*x[i];
  }

  temp = sqrt(temp);

  return temp;

}
*/

#endif

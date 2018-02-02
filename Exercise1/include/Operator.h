#ifndef _OPERATOR_H_
#define _OPERATOR_H_
#include <stdio.h>
#include <math.h>

double op=0;

double norm(double *x, int lda){

  printf("Compute norm\n");
  printf("lda = %d\n",lda);
  double temp = 0;

  for(int i=0;i<lda;i++){
    temp += x[i]*x[i];
    op += 1;
  }

  x[0] = sqrt(temp);

  return op;
}

double dot(double *x, double *y, int lda){

  printf("Compute norm\n");
  printf("lda = %d\n",lda);
  double temp = 0;

  for(int i=0;i<lda;i++){
    temp += x[i]*y[i];
  }

  return temp;
}

double mv(double *A, double *x, double *y, int lda, int col){

  printf("Compute matrix multiple vector\n");

  for(int i=0;i<col;i++){
    for(int j=0;j<m;j++){
      y[i] += A[m*i+j]*x[j];
      op += 1;
    }
  }
  printf("check answer = %f\n",y[0]);
  return op;
}

double mm(double *A, double *B, double *C, int lda, int ldb, int a_col){

  printf("Compute matrix multiple matrix\n");

  for(int i=0;i<lda;i++){
    for(int j=0;j<a_col;j++){
      for(int k=0;k<a_col;k++){
        C[j*lda+i] += A[k*lda+i]*B[k*ldb+j];
        op += 1;
      }
    }
  }

  printf("%f\n",C[0]);
  return op;

}

#endif

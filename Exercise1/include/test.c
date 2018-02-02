#include<stdio.h>
#include<stdlib.h>

int main(){

  float *A;
  int M = 10;
  A = (float *)malloc( M*sizeof( float ));

  for(int i=0;i<10;i++){
    A[i] = 1.0;
  }

  float len;
  len = (M*sizeof(A[0]))/sizeof(float);
  printf("%d\n", sizeof(A));
  printf("%d\n", sizeof(float));
  printf("%f\n",len);

  for(int i=0;i<M;i++){
    printf("%d\t",A[i]);
  }

}

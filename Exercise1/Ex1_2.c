#include <stdio.h>
#include "include/allocate.h"
#include "include/Operator.h"
#include <sys/time.h>

double wallclock(void)
{
  struct timeval tv;
  struct timezone tz;
  double t;

  gettimeofday(&tv, &tz);

  t = (double)tv.tv_sec*1000;
  t += ((double)tv.tv_usec)/1000.0;

  return t;
}

int main(){

  Init_M();

  for(int i=0;i<m*m;i++){
    A[i]=1;
    B[i]=1;
  }

  double start,end;
  start = wallclock();
  double op;
  op = mm(A,B,C,m,m,m);
  end = wallclock();

  double time = (end - start) * 1e-3;
  printf("time = %f\n",time);
  printf("%f GFLOPS\n",(op/time) * 1e-9);

  Free_M();

}

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

  //Initial A,B : matrix ; x,y : vector. (equal to 0)
  Init_V();

  printf("-------------------\n");
  printf("%d\n",m);
  for(int i=0;i<m;i++){
    x[i] = 1;
  }
  printf("-------------------\n");
  double op;

  double start, end;
  start = wallclock();
  op = norm(x,m);
  end = wallclock();

  double time = (end-start) * 1e-3;

  printf("%f GFLOPS, time = %f\n", op/time * 1e-9, time);

  Free_V();

}

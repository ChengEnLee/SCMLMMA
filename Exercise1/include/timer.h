#ifndef DGEMM_TIMER_H_
#define DGEMM_TIMER_H_

#include <stdio.h>
#include <sys/time.h>

double gettime(){

  timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sc) + double(tv.tv_usec) * 1e-6;
}

double tic( double *ptr_timer){

  *ptr_timer = gettime();
  return *ptr_timer;
}

double toc( double *ptr_timer ){

  double time = gettime() - *ptr_timer;
  printf("Elapsed time is %f seconds. \n\n", time);
  return time;
}

#endif

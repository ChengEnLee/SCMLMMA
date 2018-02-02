ALL: CPU OMP

CPU:
	gcc Matrix_Transpose.c -O3 -o trans.out
OMP:
	gcc Matrix_Transpose_omp.c -O3 -o trans_omp.out -fopenmp
clean:
	rm *.out

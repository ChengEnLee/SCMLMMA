/* 
*   This provides the framework for an SpMV GPU kernel (double precision).
*
*   The first part of the program is reading in a sparse matrix according to 
*   the example_read.c from Matrix Market: 
*   http://math.nist.gov/MatrixMarket/mmio-c.html
*   
*   In the second part, the vectors are initialized and Matrix and vectors are
*   sent to the GPU.
*   
*   The third part is the actual SpMV kernel. This is where you configure your
*   compute grid and call your kernel.
*
*   In the fourth part you validate your results. How you do this is up to you,
*   but you have to explain and document it.
*   
*   Compile this file with
*   nvcc -I./ mmio.cu spmv_framework.cu -o spmv
*   
*   If you encounter problems: hanzt@icl.utk.edu
*       
*/

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <vector>
#include <utility>  // pair
#include <mmio.h>

int M, N, nz; 
/**
    Purpose
    -------
    Returns true if first element of a is less than first element of b.
    Ignores second element. Used for sorting pairs,
    std::pair< int, magmaDoubleComplex >, of column indices and values.
*/
static bool compare_first(
    const std::pair< int, double >& a,
    const std::pair< int, double >& b )
{
    return (a.first < b.first);
}

// CPU spmv

void spmv(double *value, int *col, int *rowptr, double *x, double *y, double alpha){

    for(int ro = 0; ro < N; ro++){
        float sum = 0.0;
        for(int j = rowptr[ro]; j < rowptr[ro+1]; j++){
            sum += value[j]*x[col[j]];
        }
        y[ro] = alpha * sum;
        //if(ro < 10) printf("%f \n",y[ro]);
    }
}


// GPU spmv

__global__ void spmv_kernel(int m, int *drow, int *dcol, double *dval, double *dx, double *dy, double alpha)
{
    int count = 0;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double tmp = 0.0;
    if (tid < m) {
        for (int idx = drow[tid]; idx < drow[tid + 1]; ++idx) {
            tmp += dx[dcol[idx]] * dval[idx];
            count += 1;
        }
        dy[tid] = tmp;
    }

    __syncthreads();
    
    printf("count = %i\n", count);
    //if(ro < 10) printf("%f \n",dy[ro]);
}
  




extern "C"
int main(int argc, char *argv[])
{
    
    
    /**************************************************************************/
    /* We first read in the matrix from a .mtx file.                          */
    /**************************************************************************/
    
    int ret_code;
    MM_typecode matcode;
    FILE *f;  
    int i, *I, *J, *row, *col;
    double *val, *valt;
    std::vector< std::pair< int, double > > rowval;

    if (argc < 2)
	{
		fprintf(stderr, "SpMV routine. Usage: %s [martix-market-filename]\n", 
		        argv[0]);
		exit(1);
	}
    else    
    { 
        if ((f = fopen(argv[1], "r")) == NULL) 
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
            mm_is_sparse(matcode) )
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz);
    if (ret_code !=0)   
        exit(1);


    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    valt = (double *) malloc(nz * sizeof(double));   


    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &valt[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);
    
    /* convert the COO matrix to CSR */
    row = (int *) malloc((M+1) * sizeof(int));
    col = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));   
    
    // original code from  Nathan Bell and Michael Garland
    for ( i = 0; i < M; i++ )
        (row)[i] = 0;
    
    for ( i = 0; i < nz; i++ )
        (row)[I[i]]++;
    
    // cumulative sum the nnz per row to get row[]
    int cumsum;
    cumsum = 0;
    for( i = 0; i < M; i++ ) {
        int temp = row[i];
        (row)[i] = cumsum;
        cumsum += temp;
    }
    (row)[M] = nz;
    
    // write Aj,Ax into Bj,Bx
    for( i = 0; i < nz; i++ ) {
        int row_  = I[i];
        int dest = row[row_];
        col[dest] = J[i];
        val[dest] = valt[i];
        row[row_]++;
    }
    
    int last;
    last = 0;
    for( i = 0; i <= M; i++ ) {
        int temp  = (row)[i];
        (row)[i] = last;
        last      = temp;
    }
    
    (row)[M] = nz;

    // sort column indices within each row
    // copy into vector of pairs (column index, value), sort by column index
    for ( int k=0; k < M; ++k ) {
        int kk  = (row)[k];
        int len = (row)[k+1] - row[k];
        rowval.resize( len );
        for( i=0; i < len; ++i ) {
            rowval[i] = std::make_pair( col[kk+i], val[kk+i] );
        }
        std::sort( rowval.begin(), rowval.end(), compare_first );
        for( i=0; i < len; ++i ) {
            col[kk+i] = rowval[i].first;
            val[kk+i] = rowval[i].second;
        }
    }
    

    /**************************************************************************/
    /* Now we have the matrix in CPU main memory in CSR format.               */  
    /* The matrix A is stored as triplet A = [ row col val ].                 */
    /* We can now start with GPU allocation.                                  */
    /**************************************************************************/

    /* reserve memory for vectors x and y */
    int vecSize = N*sizeof(double);
    double *x, *y, *y2, *dx, *dy;
    x = (double*) malloc (vecSize);
    y = (double*) malloc (vecSize);
    y2 = (double*) malloc (vecSize);
    cudaMalloc( (void**) &dx, vecSize);
    cudaMalloc( (void**) &dy, vecSize);  
    
    /* reserve memory for the matrix */
    int *drow, *dcol;
    double alpha = 1.0;
    double *dval;
    
    cudaMalloc( (void**) &dval, (nz*sizeof(double)));
    cudaMalloc( (void**) &dcol, (nz*sizeof(int)));
    cudaMalloc( (void**) &drow, ((M+1)*sizeof(int)));
    
    
    /* initialize vectors and send vectors and matrix to the GPU */
    srand(time(NULL));// init random number generator
    for (i=0; i<N; i++){
        x[i] = (double) (rand()%1000);
        x[i] = 1.0;
        y[i] = 0.0;
        y2[i] = 0.0;
    }
    cudaMemcpy( dx, x, vecSize, cudaMemcpyHostToDevice );
    cudaMemcpy( dy, y, vecSize, cudaMemcpyHostToDevice );
    cudaMemcpy( dval, val, (nz*sizeof(double)), cudaMemcpyHostToDevice );
    cudaMemcpy( dcol, col, (nz*sizeof(int)), cudaMemcpyHostToDevice );
    cudaMemcpy( drow, row, ((M+1)*sizeof(int)), cudaMemcpyHostToDevice );
    
    double h_start, h_end;
    h_start = clock();
    spmv(val , col, row, x, y, 1.0);
    h_end = clock();
    printf("elapsed time by CPU : %f\n", (h_end-h_start)/CLOCKS_PER_SEC);
    /**************************************************************************/
    /* The setup for the CSR-SpMV kernel on GPU is ready. Now compute         */
    /* dy = dA* dx where dA is stored in [ drow dcol dval ].                  */  
    /**************************************************************************/   
     
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    
    // // here goes your kernel
    // // define the kernel in a seperate file, please.
    // // configure your compute grid using


    // dim3 block(256); 
    // dim3 grid((N+255)/256);
    // cudaEventRecord(start);
    // for(i=0; i<5; i++)
    //     spmv_kernel<<<grid,block>>>(M, drow, dcol, dval, dx, dy, alpha);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // // this is the runtime for 100 spmvs in ms
    // printf("runtime [ms]: %f\n", milliseconds/ 5.0 );
    // printf(" %f x\n", ((h_end-h_start))/(milliseconds/(5.0)));
    
    
    /**************************************************************************/
    /* Copy the result back and check the correctness.                        */
    /**************************************************************************/ 
    //for(int i=0; i<vecSize; i++) x[i] = y[i];
    cudaMemcpy( y, dy, vecSize, cudaMemcpyDeviceToHost);

/*
    double temp = 0.0;
    for(int i=0; i<vecSize; i++){
        temp += (x[i]-y[i])*(x[i]-y[i]);
    }
    temp = sqrt(temp);

    printf("result = %f", temp);
*/
    // How you check the correctness is up to you.
    // You can use MKL, use a CPU implementation of the SpMV, or any other 
    // validity test... 
    // You should compare the result in y with the reference result in y2
    // Explain what you do and how you do it.
    
    
    /**************************************************************************/
    /* In the end, we have to free all allocated memory.                      */
    /**************************************************************************/  
    
    //printf("val = %i, col = %i, row = %i\n", nz, N, M);
    //for(int i=0;i<10;i++) printf("%i ",col[i]);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dval);
    cudaFree(dcol);
    cudaFree(drow);
    
    free(x);
    free(y);
    free(y2);
    free(val);
    free(col);
    free(row);
  
    free(valt);
    free(I);
    free(J);
    
	return 0;
}

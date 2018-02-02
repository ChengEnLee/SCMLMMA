#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define PING 0
#define PONG 1
#define SIZE 1024*1024*1024*1.5
//#define SIZE 1024*256*25

int main(int argc, char *argv[]){

	int my_rank;
	int size,i;
	float *buffer;

	buffer = (float *)malloc( SIZE*sizeof(float));
	//for(i=0;i<SIZE;i++) buffer[i] = 0;

	double start, end;
	MPI_Status status;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if(my_rank==0) printf("start program\n");
	//MPI_Barrier(MPI_COMM_WORLD);

	start = MPI_Wtime();
	if(my_rank == 0){
		MPI_Send(buffer, SIZE, MPI_FLOAT, 1, PING, MPI_COMM_WORLD);
		MPI_Recv(buffer, SIZE, MPI_FLOAT, 1, PONG, MPI_COMM_WORLD, &status);
		//printf("Rank %d says: Ping-pong is completed.\n",my_rank);
	}
	if(my_rank == 1){
		MPI_Recv(buffer, SIZE, MPI_FLOAT, 0, PING, MPI_COMM_WORLD, &status);
		MPI_Send(buffer, SIZE, MPI_FLOAT, 0, PONG, MPI_COMM_WORLD);
		//printf("Rank %d says: Ping-pong is completed.\n",my_rank);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	//MPI_Barrier(MPI_COMM_WORLD);
	printf("Rank %d says: Ping-pong is completed.\n",my_rank);
	//fflush(stdout);
	//MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);

	int input = my_rank*2 +1;
	int result = 0;
	MPI_Reduce(&input, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	printf("Hello from rank %d in communicator with %d ranks.\n", my_rank, size);
	//fflush(stdout);
	//MPI_Barrier(MPI_COMM_WORLD);

	//if(my_rank == 0) printf("Rank 0 say : result as %i\n", result);
	printf("ssRank %d says: result is %i.\n",my_rank, result);
	//MPI_Barrier(MPI_COMM_WORLD);


	MPI_Allreduce(&input, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("qqqRank %d says: result is %i.\n",my_rank, result);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("time = %f\n", end - start);
	//MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;

}

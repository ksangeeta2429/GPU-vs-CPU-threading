#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <stdlib.h>
#include <iomanip>

#define NUM_ITER 1000000
#define NUM_OF_MSGS 5

using namespace std;

int msg_sizes[NUM_OF_MSGS] = {32, 256, 512, 1024, 2048};

int main(int argc, char *argv[]) {
	double *message;
    // Initialize the MPI execution environment
    MPI_Init(&argc, &argv);
    
    // Get the size of the group associated with a communicator 
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
	cout<<setw(20)<<"Number of messages"<<setw(15)<<"Run time (sec)"<<setw(28)<<"Bandwidth (Gbytes/sec)"<<endl;

    for(int i = 0; i < NUM_OF_MSGS; i++) {
	    int msgSize = msg_sizes[i];

/*	    try {
	    	message = new double(msgSize);
		}catch (std::bad_alloc&) {
			cout<<"ERROR: Memory allocation failed for message!\n";
		    MPI_Finalize();
		    exit(1);
		}*/

		// Allocate the memory for the messages
	    double *message = (double *) malloc(sizeof(double) * msgSize);

            // Quit if the message allocation failed
	    if(!message) {
	    	cout<<"ERROR: Memory allocation failed for message!\n";
		    MPI_Finalize();
		    exit(1);
	    }

	    MPI_Barrier(MPI_COMM_WORLD);
	    double start = MPI_Wtime();

        if(rank == 0) {
	    	for(int j = 0; j < NUM_ITER; j++) {
				MPI_Send(message, msgSize, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(message, msgSize, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(message, msgSize, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(message, msgSize, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
	    } else if(rank == 1){
	    	for(int j = 0; j < NUM_ITER; j++) {
				MPI_Recv(message, msgSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(message, msgSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
				MPI_Recv(message, msgSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(message, msgSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
	    }

	    double end = MPI_Wtime();
	    double elapsed_time = end - start;
	    double avg_time = elapsed_time/(4 * NUM_ITER);
	    double bandwidth = (sizeof(double) * msgSize) / avg_time;

	    if(rank == 0)
	    	cout<<setw(20)<<right<<msgSize<<setw(15)<<right<<elapsed_time<<setw(28)<<right<<setprecision(4)<<(bandwidth / 10e9);

	    cout<<endl;
	    
	    free(message);	    
    }//end of for
        
    //Terminate the MPI execution environment 
    MPI_Finalize();

    return 0;
}

//
// Created by Sowmya Parameshwara on 23/10/16.
//

/**
 *
Serial code which was parallelised :
Loop 1 : for (norm = 0; norm < N - 1; norm++) {
            Loop 2 : for (row = norm + 1; row < N; row++) {
                                     multiplier = A[row][norm] / A[norm][norm];
                      Loop 3 : for (col = norm; col < N; col++) {
                                            A[row][col] -= A[norm][col] * multiplier;
                                     }
                                     B[row] -= B[norm] * multiplier;
                           }
                }


 *  1) There exists a data dependency across iterations of Loop 1, hence it was not parallelised.
 *  2) There is no data dependency across iterations of Loop 2, hence it was parallelised using data parallelism.
 *  3) Process with rank_id 0 is responsible for input and output.
 *  4) For Each iteration of outer loop, following steps :
 *      a) Process 0 will broadcast a[outerloopindex][0] and b[outerloopindex].
 *      b) Process 0 will than send asynchronously(MPI_Isend) to each process,a[row] and b[row] corresponding to the iteration it has to compute using static interleaving logic.
 *         (MPI_Isend is better performing then MPI_Send).
 *      c) Process 0 will compute the iteration assigned to it.
 *      d) Process 0 will wait for the asynchronous send of all processors to be complete. (This improves performance, by overlapping computation time with send time.)
 *      e) Process 0 will recieve(MPI_Recv) from each processor the computer result.
 *
 *      f)Other processors will recieve(MPI_Recv) from processor 0 the iterations assigned to them.
 *      g)They will compute the iteration assigned to it.
 *      h)They will synchronously send(MPI_Send) the result of their computation to processor 0.
 *
 *      i)A barrier is now kept, once all processors complete, next iteration continues.
 *  5) Processor 0 will now perform backsubstitution().
 *  6) Processor 0 will output the result.
 *
 *  This algorithm scales assuming N is much larger than the number of processes.
 *
 *  Steps to compile and execute :
 *  1)  mpicc -c GaussMPI.c
 *  2)  mpicc -o GaussMPI GaussMPI.o
 *  3)  create bash file : vi run_gauss_mpi.bash
 *  4) File contents :
 *      #!/bin/bash
        mpirun -npernode 8 ./GaussMPI 20 4   <Argument 1 : Size of matrix, Argument 2 : Random seed value>
 *  5) Run : qsub -pe mpich 1 run_gauss_mpi.bash
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>

#define MAXN 20000  /* Max value of N */
int N;
int my_rank;   /* My process rank           */
int p;         /* The number of processes   */
/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];


void parameters(int argc, char **argv);
void initializeInputs();
void printInputs();
void printX();
void backSubstitution();
void gaussMPI();



int main(int argc, char **argv) {


    double startWTime,endWTime;

    /* MPI Initialise */

    MPI_Init(&argc, &argv);

    /* Get my process rank */

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many processes are being used */

    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /** Processor 0 performs input operation */
    if(my_rank == 0 ){
        /* Process program parameters */
        parameters(argc, argv);
        /* Initialize A and B */
        initializeInputs();
        /* Print input matrices */
        printInputs();
        /**Get time before starting guassian algorithm */
        startWTime = MPI_Wtime();


    }

    /* Guassian threadP algorithm*/
     gaussMPI();

    /**Processor 0 will perform output*/
    if(my_rank == 0 ) {
        endWTime = MPI_Wtime();

        /* Backward substitution on row echelon form matrix to determine cooefficients*/
        backSubstitution();
        /* Display output */
        printX();

        /* Display timing results */
        printf("\nTotal time = %f .\n", (endWTime - startWTime));

    }

    /*MPI Finalize*/
    MPI_Finalize();
    exit(0);

}




/* Set the program parameters from the command-line arguments
 * argv[1] : Matrix Size
 * argv[2] : Random Seed
 * r*/
void parameters(int argc, char **argv) {
    int seed = 0;
    if(argc == 3){
        seed = atoi(argv[2]);
        srand(seed);

        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    } else {
        printf("Usage: %s <matrix_dimension> [random seed]  \n",
               argv[0]);
        exit(0);
    }
    /* Print parameters */
    printf("\nMatrix dimension N = %i. Seed = %d .\n", N,seed);
}

/* Initialize A and B (and X to 0.0s) */
void initializeInputs() {
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }

}

/* Print input matrices A[N][N] and B[N] generated by randomiser */
void printInputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < N; col++) {
            printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
        }
    }
}

/**
 * Prints X coefficient values
 */
void printX() {
    int row;
    if (N < 100) {
        printf("\nX = [");
        for (row = 0; row < N; row++) {
            printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
        }
    }
}



/**
 * Use backward substitution technique to calculate X values.
 * Say X0, X1, X2 .....XN needs to be calculated. This method starts calculating XN and on completion
 * uses this value to calculate X(N-1) and so on goes backwards.
 */
void backSubstitution(){
    int norm, row, col;
    /* Back substitution */
    for (row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N-1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }

}


/**
 * Gaussian elimination without pivoting and static interleaving using MPI.
 */
void gaussMPI() {
    int norm, row, col,i;
    float multiplier;


    /**Processor 0 broadcasts N to all other processors.*/
    MPI_Bcast(&N, 1, MPI_INT,0,MPI_COMM_WORLD);

    for (norm = 0; norm < N - 1; norm++) {

        /**Processor 0 broadcasts the outer loop row being processed to all other processors*/
        MPI_Bcast(&A[norm][0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


        if(my_rank==0){
            MPI_Status status1[N];
            MPI_Status status2[N];
            MPI_Status status3,status4;
            MPI_Request request1[N];
            MPI_Request request2[N];
            for( i=1; i<p; i++){
                for (row = norm + 1 + i; row < N; row += p) {
                    /*Asynchronously send data to be computed to the respective processor*/
                    MPI_Isend(&A[row], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request1[row]);
                    MPI_Isend(&B[row], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request2[row]);
                }
            }

            /**
             * Processor 0 computes the inner loop assigned to it.
             */
            for (row = norm + 1; row < N; row += p) {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }

            /**
             * Processor 0 waits for it's previos send operation to complete, after which it will recieve computed data from other processes.
             */
            for (i = 1; i < p; i++) {
                for (row = norm + 1 + i; row < N; row += p) {
                    MPI_Wait(&request1[row], &status1[row]);
                    MPI_Wait(&request2[row], &status2[row]);
                    MPI_Recv(&A[row], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status3);
                    MPI_Recv(&B[row], 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status4);
                }
            }
        }else{
            MPI_Status status1;
            MPI_Status status2;

            /**
             * All other processor will recieve the iterations it needs to compute, perform the processing and than send the result to processor 0.
             */
            for (row = norm + 1 + my_rank; row < N; row += p) {
                MPI_Recv(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status1);
                MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status2);
                /*Gaussian elimination*/
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
                /*Send back the results*/
                MPI_Send(&A[row], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);// Wait till other processors complete, and than carry on to next iteration.
    }


}



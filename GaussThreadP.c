//
// Created by Sowmya Parameshwara on 9/30/16.
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
 *  3) There can be 2 cases in the system, we have enough threads as the number of rows to be processed, we have lesser threads than the number of rows to be processed.
 *  4) In case 1 each thread processes one iteration only.
 *  5) In case 2 (This case ensures that the code can scale), a single thread may handle multiple iterations of Loop2.Each thread will start the iteration with ‘threadId’
 *     index and then pick all iterations which are at an offset of “numberOfThreads’ from their current index.
 *  6) Numer of processors and number of threads allowed per processor are both taken as command line argument.
 *  7) General idea, In first iteration of loop 1 we need to initialise all entries in column[0] starting from row[1] to ‘0’. This happens in loop 2
 *     by “Add to one row a scalar multiple of another” such that corresponding column value becomes 0. This operation happens in parallel where all
 *     the threads performs this process on different rows. After this is done, the next column will be picked up in loop 1 and the operation is repeated.
 *     Once the matrix is in row echelon form, backsubsitution is performed to calculate the result.
 *
 *  Steps to compile and execute :
 *  1) gcc -o GaussThreadP GaussThreadP.c -lpthread -mcmodel=medium
 *  2) ./GaussThreadP 10 4 4 1   <Argument 1 : Size of matrix, Argument 2 : Random seed value, Argument 3 : Number of processors , Argument 4 : Number of threads per processor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>
#include <pthread.h>

#define MAXN 20000  /* Max value of N */
int numberOfProcessors, numberOfThreadsPerProcessor, N, totalNumberOfThreads;
/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];

/**Struct to hold threadid and outerrow value which will be passed to the parallelised code block*/
struct ThreadParam{
    int threadId;
    int outerRow;
};


void parameters(int argc, char **argv);
void initializeInputs();
void printInputs();
void printX();
void gaussThreadP();
void gaussThreadPEnoughThreadsToProcessAllRows();
void gaussThreadPLesserThreadsToProcessAllRows();
void *rowFactorMultiplicationWithoutSkipLogic(struct ThreadParam * threadParam);
void *rowFactorMultiplicationWithSkipLogic(struct ThreadParam * threadParam);
void backSubstitution();
void gauss();



int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval etStart, etStop;  /* Elapsed times using gettimeofday() */
    struct timezone dummyTz;
    unsigned long long startTime, endTime;

    /* Process program parameters */
    parameters(argc, argv);
    /* Initialize A and B */
    initializeInputs();
    /* Print input matrices */
    printInputs();
    /**Get time before starting guassian algorithm */
    gettimeofday(&etStart, &dummyTz);
    /* Guassian threadP algorithm*/
    //gauss();
    gaussThreadP();
    /**Get time after completing guassian algorithm*/
    gettimeofday(&etStop, &dummyTz);
    /* Display output */
    printX();
    startTime = (unsigned long long)etStart.tv_sec * 1000000 + etStart.tv_usec;
    endTime = (unsigned long long)etStop.tv_sec * 1000000 + etStop.tv_usec;
    /* Display timing results */
    printf("\nElapsed time = %g ms.\n",(float)(endTime - startTime)/(float)1000);

    exit(0);

}


/* Set the program parameters from the command-line arguments
 * argv[1] : Matrix Size
 * argv[2] : Random Seed
 * argv[3] : Number of processors
 * argv[4] : Number of threads per processor*/
void parameters(int argc, char **argv) {
    int seed = 0;
    if(argc == 5){
        numberOfProcessors = atoi(argv[3]);
        numberOfThreadsPerProcessor = atoi(argv[4]);
        seed = atoi(argv[2]);
        srand(seed);

        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }else if(argc == 3){
        seed = atoi(argv[2]);
        srand(seed);

        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
        numberOfProcessors = N;
        numberOfThreadsPerProcessor = 1;
    } else {
        printf("Usage: %s <matrix_dimension> [random seed] [number of processors] [number of threads per processors] \n",
               argv[0]);
        exit(0);
    }
    /* Print parameters */
    printf("\nMatrix dimension N = %i. Number of processors = %d . Number of threads per processor = %d . Seed = %d .\n", N,numberOfProcessors,numberOfThreadsPerProcessor,seed);
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
 *  It determines the number of threads.
 *  If number of threads are equal to the number of rows to be processed it follows strategy : gaussThreadPEnoughThreadsToProcessAllRows
 *  Else it follows : gaussThreadPLesserThreadsToProcessAllRows
 */
void gaussThreadP(){
    int norm, row, col;  /* Normalization row, and zeroing element row and col */
    float multiplier;
    //If number of threads is greater than number of rows to process than threadcount is equal to number of rows else it is equal
    //to the number of threads available.
    totalNumberOfThreads = ((N-1)>(numberOfProcessors*numberOfThreadsPerProcessor))?(numberOfProcessors*numberOfThreadsPerProcessor):(N-1);
    if(totalNumberOfThreads == N-1){
        gaussThreadPEnoughThreadsToProcessAllRows(); //logic for guassian elimination when number of threads equals number of rows to process.
    }else{
        gaussThreadPLesserThreadsToProcessAllRows(); //logic for guassian elimination when number of threads is less than number of rows to process.
    }

    /* Backward substitution on row echelon form matrix to determine cooefficients
     */
    backSubstitution();
}

/**
 * It creates and starts thread. It passes the outerrow and threadindex value to the parallelfunction.
 * It also as a loop for calling join on threads.
 */
void gaussThreadPEnoughThreadsToProcessAllRows(){
    printf(" Parallel computing, number of threads equal to maximum number of rows to be processed \n");
    pthread_t pthreads[totalNumberOfThreads];
    struct ThreadParam* param = malloc(totalNumberOfThreads* sizeof(struct ThreadParam));
    int outerRow;
    for(outerRow = 0 ; outerRow < N-1; outerRow++){
        int threadIndex;
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            param[threadIndex].threadId = threadIndex;
            param[threadIndex].outerRow = outerRow;
            pthread_create(&pthreads[threadIndex],NULL,rowFactorMultiplicationWithoutSkipLogic,&param[threadIndex]);
        }
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            pthread_join(pthreads[threadIndex],NULL);
        }

    }
    free(param);
}

/**
 * It creates and starts thread. It passes the outerrow and threadindex value to the parallelfunction.
 * It also as a loop for calling join on threads.
 */
void gaussThreadPLesserThreadsToProcessAllRows(){
    printf("Parallel computing, number of threads lesser than maximum number of threads to be processed. \n ");
    pthread_t pthreads[totalNumberOfThreads];
    struct ThreadParam *param = malloc(totalNumberOfThreads* sizeof(struct ThreadParam));

    int outerRow;
    for(outerRow = 0 ; outerRow < N-1; outerRow++){
        int threadIndex;
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            param[threadIndex].threadId = threadIndex;
            param[threadIndex].outerRow = outerRow;
            pthread_create(&pthreads[threadIndex],NULL,rowFactorMultiplicationWithSkipLogic,&param[threadIndex]);
        }
        for(threadIndex = 0 ;threadIndex < totalNumberOfThreads; threadIndex++){
            pthread_join(pthreads[threadIndex],NULL);
        }

    }
    free(param);
}

/**
 * Since number of threads are equal to the maximum number of rows to be processed. In each function call
 * we apply row reduction for only one row,if the row index is valid.
 * This row is determined by the threadid and outerrow index.
 * @param threadParam
 * @return
 */
void *rowFactorMultiplicationWithoutSkipLogic(struct ThreadParam * threadParam){
    int col;
    float multiplier;
    int outerRow = threadParam->outerRow;
    int innerRow = threadParam->threadId+outerRow+1;
    if(innerRow < N) {
        multiplier = A[innerRow][outerRow] / A[outerRow][outerRow];
        for (col = outerRow; col < N; col++) {
            A[innerRow][col] -= A[outerRow][col] * multiplier;
        }
        B[innerRow] -= B[outerRow] * multiplier;
    }
}


/**
 * Since number of threads are lesser than the maximum number of rows to be processed. In each function call
 * we apply row reduction to multiple rows. We calculate the startIndex where the row reduction should be applied using
 * threadId. We than perform row reduction skipping rowIndex by numberOfThreads until we reach the end.
 * @param threadParam
 * @return
 */
void *rowFactorMultiplicationWithSkipLogic(struct ThreadParam * threadParam){
    int innerRow, col;
    float multiplier;
    int startIndex = threadParam->threadId+1;
    int outerRow = threadParam->outerRow;
    for (innerRow = startIndex+outerRow; innerRow < N; innerRow+=totalNumberOfThreads) {
        multiplier = A[innerRow][outerRow] / A[outerRow][outerRow];
        for (col = outerRow; col < N; col++) {
            A[innerRow][col] -= A[outerRow][col] * multiplier;
        }
        B[innerRow] -= B[outerRow] * multiplier;
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





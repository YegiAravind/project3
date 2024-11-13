#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 2048        // Matrix dimension
#define BLOCK_SIZE 64    // Block size for optimized multiplication
#define FACTOR 1.1       // Factor for initialization

double matrixA[SIZE][SIZE] = {0.0};
double matrixB[SIZE][SIZE] = {0.0};
double resultMatrix[SIZE][SIZE] = {0.0};

// Function to initialize matrices with random values
void initializeMatrix() {
    printf("Initializing matrices...\n");
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            srand(i + j);  // Seed based on index for unique values
            matrixA[i][j] = (rand() % 10) * FACTOR;
            matrixB[i][j] = (rand() % 10) * FACTOR;
        }
    }
    printf("Initialization complete.\n");
}

// Function for block-optimized parallel matrix multiplication
void multiplyMatricesBlockOptimized() {
    printf("Starting block-optimized matrix multiplication...\n");

    #pragma omp parallel for
    for (int i = 0; i < SIZE; i += BLOCK_SIZE) {
        for (int j = 0; j < SIZE; j += BLOCK_SIZE) {
            for (int k = 0; k < SIZE; k += BLOCK_SIZE) {
                // Multiply sub-blocks
                for (int ii = i; ii < i + BLOCK_SIZE && ii < SIZE; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < SIZE; jj++) {
                        double sum = 0.0;
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < SIZE; kk++) {
                            sum += matrixA[ii][kk] * matrixB[kk][jj];
                        }
                        resultMatrix[ii][jj] += sum;
                    }
                }
            }
        }
    }
    printf("Block-optimized matrix multiplication complete.\n");
}

int main() {
    printf("Matrix Multiplication Program\n");
    printf("Matrix size: %d x %d\n", SIZE, SIZE);
    printf("Block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);

    // Initialize matrices
    initializeMatrix();

    // Measure execution time for block-optimized parallel matrix multiplication
    clock_t start = clock();
    multiplyMatricesBlockOptimized();
    clock_t end = clock();

    // Calculate and print execution time
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for block-optimized parallel matrix multiplication: %f seconds\n", timeTaken);

    // Optionally, print a part of the result matrix to verify computation
    printf("Sample result (top-left 3x3 submatrix):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.2f ", resultMatrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}

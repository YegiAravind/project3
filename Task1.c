#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 2048
#define FACTOR 1.1

double matrixA[SIZE][SIZE] = {0.0};
double matrixB[SIZE][SIZE] = {0.0};
double resultMatrix[SIZE][SIZE] = {0.0};

void initializeMatrix() {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            srand(i + j);
            matrixA[i][j] = (rand() % 10) * FACTOR;
            matrixB[i][j] = (rand() % 10) * FACTOR;
        }
    }
}

void multiplyMatrices() {
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            double sum = 0.0;
            for (int k = 0; k < SIZE; k++) {
                sum += matrixA[i][k] * matrixB[k][j];
            }
            resultMatrix[i][j] = sum;
        }
    }
}

int main() {
    initializeMatrix();

    clock_t start = clock();
    multiplyMatrices();
    clock_t end = clock();

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for parallel matrix multiplication: %f seconds\n", timeTaken);

    return 0;
}

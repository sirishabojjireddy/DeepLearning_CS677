#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include "dproductkernel.h"

int main(int argc, char* argv[]) {

	/* Initialize variables */
        FILE *fp;
        size_t size;
        unsigned int rows= atoi(argv[3]);
        unsigned int cols= atoi(argv[4]);
        int CUDA_DEVICE = atoi(argv[5]);
        int THREADS = atoi(argv[6]);
	//printf("Rows: %d\nCols: %d\nDEVICE: %d\nTHREADS: %d\n", rows, cols, CUDA_DEVICE, THREADS);

	/* Declare and Malloc Host Variables */
        size = (size_t)((size_t)rows * (size_t)cols);
	//printf("Size of data: %u\n", size);
        int BLOCKS;
        float *dataT = (float*)malloc((size_t)size * sizeof(float));
        float *dataV = (float*)malloc((size_t)cols * sizeof(float));
        float *host_results = (float*)malloc((size_t)rows * sizeof(float));
        unsigned int jobs;

	/* Declare Kernel Variables */
        float *dev_dataT;
	float *dev_dataV;
        float *results;

	/* Store data from files to host variables */
        fp = fopen(argv[1], "r");
        if (fp == NULL) {
                printf("Cannot Open the File");
                return 0;
        }

        if(dataT == NULL) {
                printf("ERROR: Memory for data not allocated.\n");
	}

    	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++) {
               		fscanf(fp, "%f", &dataT[(i * cols) + j]);
           	}
    	}
	fclose(fp);
	//printf("Data Read Successfully!!!\n");

        fp = fopen(argv[2], "r");
        if (fp == NULL) {
                printf("Cannot Open the File");
                return 0;
        }

        if(dataV == NULL) {
                printf("ERROR: Memory for data not allocated.\n");
        }

	for (int i = 0; i < cols; i++) {
        	fscanf(fp, "%f", &dataV[i]);
	}

        fclose(fp);
	//printf("W Vector Read Successfully\n");
	
	fflush(stdout);
	/* cudaMalloc Kernel Variables */
	cudaError err = cudaSetDevice(CUDA_DEVICE);
        if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

        err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float));
        if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }

        err = cudaMalloc((float**) &dev_dataV, (size_t) cols * (size_t) sizeof(float));
        if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }

        err = cudaMalloc((float**) &results, (size_t) rows * sizeof(float));
        if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
	
	/* Copy Host Variables to Kernel Variables */
        err = cudaMemcpy(dev_dataT, dataT, (size_t)size * (size_t)sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }

        err = cudaMemcpy(dev_dataV, dataV, (size_t)cols * (size_t)sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }

        jobs = rows;
        BLOCKS = (jobs + THREADS - 1)/THREADS;
	fflush(stdout);

	/* Kernel Function Call */
	dproductkernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,dev_dataV,results);

	/* Copy the results back to Host Memory */
        err = cudaMemcpy(host_results, results, rows * sizeof(float), cudaMemcpyDeviceToHost);
	if(err != cudaSuccess) { printf("Error copying data from GPU\n"); }

	/* Print the results */
        for(int k = 0; k < jobs; k++) {
             printf("%f \n", host_results[k]);
        }
	printf("\n");
	/* Free Cuda Memory */
        cudaFree( dev_dataT );
	cudaFree( dev_dataV );
        cudaFree( results );

        return 0;
}

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "dproductkernel.h"

__global__ void dproductkernel(unsigned int rows, unsigned int cols, float *mdata, float *vdata, float *results){
	int tid  = threadIdx.x + blockIdx.x * blockDim.x;
	float dp  = 0;
	if(tid < rows) {
 		for(int j = 0; j < cols; j++) {
			dp += vdata[j]*mdata[tid*cols + j];
		}
       		results[tid] = dp;
	}
}

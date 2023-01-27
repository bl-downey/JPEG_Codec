#pragma once
//includes 
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

// cuda error check
inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}

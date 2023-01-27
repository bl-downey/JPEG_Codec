#ifndef CUDACC
#define CUDACC
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <math.h>
#include <stdio.h>
#include <list>

#define pi_math 3.142857

#define BLOCK_WIDTH 8
#define BLOCK_SIZE BLOCK_WIDTH*BLOCK_WIDTH


/*
	DCT Kernel
	input  -> unsigned char* img_in is the input image data
	output -> float* dct_out is the transformed output data
	
	int m is the number of rows
	int n is the number of cols	
	both m and n are default 8 for 8x8 DCT
*/
__global__ void dct_kernel1(unsigned char* img_in, float* dct_out, int m=8, int n=8)
{
	// global row and col
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int k, l;
	float ci, cj, dct1, sum;

	// boundary check
	if (row < m && col < n) {
		// theory: if row is row 0 of current block
		if (row % 8 == 0)
			ci = 1 / sqrtf(8);
		else
			ci = sqrtf(2) / sqrtf(8);
		// theory: if col is col 0 of current block
		if (col % 8 == 0)
			cj = 1 / sqrtf(8);
		else
			cj = sqrtf(2) / sqrtf(8);

		sum = 0;
		// theory: start in the top left corner of the 8x8 block
		int startIdx_row = row - (row % 8);
		int startIdx_col = col - (col % 8);
		for (k = startIdx_row; k < startIdx_row+8; k++) {
			for (l = startIdx_col; l < startIdx_col+8; l++) {
				dct1 = int(img_in[k*n+l]) *
						cosf((2*(k%8)+1)*(row%8)* pi_math / (2*8)) *
						cosf((2*(l%8)+1)*(col%8)* pi_math / (2*8));
				sum += dct1;
			}
		}
		dct_out[row*n+col] = ci * cj * sum;
	}
}

/*
	Quantization Kernel
	
	input  -> float *dct_out is the transformed img data to be quantized
		   -> int* gmem_q GMEM based q matrix, load into shared matrix upon kernel call

	output -> int *quant_out is quantized image output of the quant function		
*/
__global__ void quant_kernel1(float* dct_out, int* quant_out, int* gmem_q, int m, int n)
{
	// q shared mem
	__shared__ int ds_q[BLOCK_SIZE];

	// tx and ty and row/col
	int tx = threadIdx.x; int ty = threadIdx.y; 
	int row = blockDim.y * blockIdx.y + ty;
	int col = blockDim.x * blockIdx.x + tx;

	// load shared mem for q
	if (tx < BLOCK_WIDTH && ty < BLOCK_WIDTH) {		
		ds_q[tx * BLOCK_WIDTH + ty] = gmem_q[tx * BLOCK_WIDTH + ty];
	}

	// required syncthreads - prevent invalid shared memory accesses
	__syncthreads();

	// check bounds
	if (row < m && col < n) {
		quant_out[row * n + col] = dct_out[row * n + col] / ds_q[(row % 8) * BLOCK_WIDTH + (col % 8)];
	}
}

/*
	Zig Zag Scan Kernel
	input  -> int *quant_out image to be scanned in 8x8 blocks
	output -> int *zigzag_out is the 64-d output array of scan
	
	int m is the number of rows
	int n is the number of cols	
	both m and n are default 8 for 8x8 zig zag scan
	
	Adaptation of https://www.geeksforgeeks.org/print-matrix-in-zig-zag-fashion/
*/
__global__ void zigzag_kernel1(int* quant_out, int* zigzag_out, int m_bound = 8, int n_bound = 8)
{
	// thread block local row col values
	int tx = threadIdx.x; int ty = threadIdx.y;
	int init_row = blockDim.y * blockIdx.y + ty;
	int init_col = blockDim.x * blockIdx.x + tx;

	int row = init_row * 8;
	int col = init_col * 8;

	//printf("init_row, init_col: (%d,%d)\trow, col: (%d, %d)\n", init_row, init_col, row, col);

	// fill out array index
	int idx_out = 0;

	// flag for incremening r or c in iterations of reading scan
	bool row_inc = false;

	// define size of block to be zig zagged
	int m = 8; int n = 8;

	int mult_factor = sqrtf(m_bound * n_bound / 64);

	//printf("mult_factor : %d\n", mult_factor);

	// nested loop to compute lower half of zig-zag pattern
	int mn = min(m, n);
	if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
		//printf("row,col : (%d, %d)\n", row, col);
		for (int len = 1; len <= mn; ++len) {
			for (int i = 0; i < len; ++i) {
				
				zigzag_out[(mult_factor * init_row + init_col) * 64 + idx_out] = quant_out[row * n_bound + col];
				idx_out++;

				if (i + 1 == len)
					break;
				// If row_increment value is true
				// increment row and decrement col
				// else decrement row and increment
				// col
				if (row_inc)
					++row, --col;
				else
					--row, ++col;
			}

			if (len == mn)
				break;

			// Update row or col value according
			// to the last increment
			if (row_inc)
				++row, row_inc = false;
			else
				++col, row_inc = true;
		}
	}
 
    // Update the indexes of row and col variable
    if ((row % 8) == 0) {
        if ((col % 8) == m - 1)
            ++row;
        else
            ++col;
        row_inc = 1;
    }
    else {
        if ((row % 8) == n - 1)
            ++col;
        else
            ++row;
        row_inc = 0;
    }
 
    // Print the next half zig-zag pattern
    int MAX = max(m, n) - 1;
    for (int len, diag = MAX; diag > 0; --diag) {
 
        if (diag > mn)
            len = mn;
        else
            len = diag;
 
        for (int i = 0; i < len; ++i) {
			if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
				zigzag_out[(mult_factor * init_row + init_col) * 64 + idx_out] = quant_out[row * n_bound + col];
				idx_out++;

				if (i + 1 == len)
					break;

				// Update row or col value according
				// to the last increment
				if (row_inc)
					++row, --col;
				else
					++col, --row;
			}
        }
 
        // Update the indexes of row and col variable
        if ((row%8) == 0 || (col % 8) == m - 1) {
            if ((col % 8) == m - 1)
                ++row;
            else
                ++col;
 
            row_inc = true;
        }
 
        else if ((col % 8) == 0 || (row % 8) == n - 1) {
            if ((row % 8) == n - 1)
                ++col;
            else
                ++row;
 
            row_inc = false;
        }
    }
}

/*
	Run Length Encoding Kernel
	input  -> int *zigzag_out is the 64-d array to be encoded
	output -> int* encoded is an array storing the value and then the length, repeated
		
	Adaptation of https://www.geeksforgeeks.org/run-length-encoding/
*/
__global__ void rle_kernel1(int* zigzag_out, int* encoded_val, int* encoded_rep, int m_bound, int n_bound)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int mult_factor = sqrtf(m_bound * n_bound / 64);

	int save_idx = -1;
	if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
		for (int i = 0; i < 64; i++) {

			// Count occurrences of current character
			int count = 1;
			while (i < 64 - 1 && zigzag_out[(mult_factor * row + col) * 64 + i] == zigzag_out[(mult_factor * row + col) * 64 + i + 1]) {
				count++;
				i++;
			}
			if (count > 0)
				save_idx++;
			encoded_val[(mult_factor * row + col) * 64 + save_idx] = zigzag_out[(mult_factor * row + col) * 64 + i];
			encoded_rep[(mult_factor * row + col) * 64 + save_idx] = count;

		}
	}
}

/*
	Run Length Encoding Kernel
	input  -> std::list<int>& rle_compressed is the compressed list representation
	output -> int *zigzag_out is the reconstructed output of zig zag
*/
__global__ void irle_kernel1(int* encoded_val, int* encoded_rep, int* zigzag_out, int m_bound, int n_bound)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int mult_factor = sqrtf(m_bound * n_bound / 64);

	// reconstruct the zig zag out based on two array representation
	int save_idx = 0;
	int v; int rep;
	if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
		for (int i = 0; i < 64; i++) {
			v = encoded_val[(mult_factor * row + col) * 64 + i];
			rep = encoded_rep[(mult_factor * row + col) * 64 + i];
			for (int j = 0; j < rep; j++) {
				zigzag_out[(mult_factor * row + col) * 64 + save_idx + j] = v;
			}
			save_idx += rep;
		}
	}
}

/*
	Inverse Zig Zag Kernel
	input  -> int *zigzag_out is the decompressed zig zag out 
	output -> int *izigzag_out is the reconstructed img
	
	int m is the number of rows
	int n is the number of cols	
	both m and n are default 8 for 8x8 inverse zig zag scan
		
*/
__global__ void izigzag_kernel1(int *izigzag_out, int *reconstructed_block, int m_bound=8, int n_bound=8)
{
	// thread block local row col values
	int tx = threadIdx.x; int ty = threadIdx.y;
	int init_row = blockDim.y * blockIdx.y + ty;
	int init_col = blockDim.x * blockIdx.x + tx;

	int row = init_row * 8;
	int col = init_col * 8;

	// fill out array index
	int idx_out = 0;

	// flag for incremening r or c in iterations of reading scan
	bool row_inc = false;

	// define size of block to be zig zagged
	int m = 8; int n = 8;

	int mult_factor = sqrtf(m_bound * n_bound / 64);

	// nested loop to compute lower half of zig-zag pattern
	int mn = min(m, n);
	for (int len = 1; len <= mn; ++len) {
		for (int i = 0; i < len; ++i) {
			if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
				reconstructed_block[row * n_bound + col] = izigzag_out[(mult_factor * init_row + init_col) * 64 + idx_out];
				idx_out++;

				if (i + 1 == len)
					break;
				// If row_increment value is true
				// increment row and decrement col
				// else decrement row and increment
				// col
				if (row_inc)
					++row, --col;
				else
					--row, ++col;
			}
		}

		if (len == mn)
			break;

		// Update row or col value according
		// to the last increment
		if (row_inc)
			++row, row_inc = false;
		else
			++col, row_inc = true;
	}

	// Update the indexes of row and col variable
	if ((row % 8) == 0) {
		if ((col % 8) == m - 1)
			++row;
		else
			++col;
		row_inc = 1;
	}
	else {
		if ((row % 8) == n - 1)
			++col;
		else
			++row;
		row_inc = 0;
	}

	// Print the next half zig-zag pattern
	int MAX = max(m, n) - 1;
	for (int len, diag = MAX; diag > 0; --diag) {

		if (diag > mn)
			len = mn;
		else
			len = diag;

		for (int i = 0; i < len; ++i) {
			if (row > -1 && row < m_bound && col > -1 && col < n_bound) {
				reconstructed_block[row * n_bound + col] = izigzag_out[(mult_factor * init_row + init_col) * 64 + idx_out];
				idx_out++;

				if (i + 1 == len)
					break;

				// Update row or col value according
				// to the last increment
				if (row_inc)
					++row, --col;
				else
					++col, --row;
			}
		}

		// Update the indexes of row and col variable
		if ((row % 8) == 0 || (col % 8) == m - 1) {
			if ((col % 8) == m - 1)
				++row;
			else
				++col;

			row_inc = true;
		}

		else if ((col % 8) == 0 || (row % 8) == n - 1) {
			if ((row % 8) == n - 1)
				++col;
			else
				++row;

			row_inc = false;
		}
	}
}

/*
	Inverse Quantization Kernel
	input  -> int *izigzag_out is the decompressed zig zag out 
	output -> int *iquant_out is the inverse quantized decompressed output
*/
__global__ void iquant_kernel1(int *izigzag_out, int *iquant_out, int* gmem_q, int m, int n)
{
	// q shared mem
	__shared__ int ds_q[BLOCK_SIZE];

	// tx and ty and row/col
	int tx = threadIdx.x; int ty = threadIdx.y;
	int row = blockDim.y * blockIdx.y + ty;
	int col = blockDim.x * blockIdx.x + tx;

	// load shared mem for q
	if (tx < BLOCK_WIDTH && ty < BLOCK_WIDTH) {
		//printf("tx,tx: (%d,%d)\tq mem = %d\n", tx, ty, gmem_q[tx * BLOCK_WIDTH + ty]);
		ds_q[tx * BLOCK_WIDTH + ty] = gmem_q[tx * BLOCK_WIDTH + ty];
	}

	// required syncthreads so that threads out of the above bounds cannot continue until shared mem is full
	__syncthreads();

	// check bounds
	if (row < m && col < n) {
		/*if (izigzag_out[row * n + col] * ds_q[(row % 8) * BLOCK_WIDTH + (col % 8)] != 0)
			printf("row, col: (%d, %d)\n val = %d\n ds_q = %d\n", row, col, izigzag_out[row * n + col] * ds_q[(row % 8) * BLOCK_WIDTH + (col % 8)], ds_q[(row % 8) * BLOCK_WIDTH + (col % 8)]);*/
		// implicit conversion from dct float to int with mult by int
		iquant_out[row * n + col] = izigzag_out[row * n + col] * ds_q[(row % 8) * BLOCK_WIDTH + (col % 8)];
	}
}

/*
	Inverse DCT Kernel
	input  -> int *iquant_out is the inverse quantized results
	output -> unsigned char* recon_img is the inverse dct, reconstructed image
*/
__global__ void idct_kernel1(int* iquant_out, unsigned char* recon_img, int m=8, int n=8)
{
	// global row and col
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int k, l;
	float ck, cl, dct1, sum;

	// boundary check
	if (row < m && col < n) {

		sum = 0;
		// theory: start in the top left corner of the 8x8 block
		int startIdx_row = row - (row % 8);
		int startIdx_col = col - (col % 8);
		for (k = startIdx_row; k < startIdx_row + 8; k++) {
			for (l = startIdx_col; l < startIdx_col + 8; l++) {
				// theory: if row is row 0 of current block
				if (k % 8 == 0)
					ck = 1 / sqrtf(8);
				else
					ck = sqrtf(2) / sqrtf(8);
				// theory: if col is col 0 of current block
				if (l % 8 == 0)
					cl = 1 / sqrtf(8);
				else
					cl = sqrtf(2) / sqrtf(8);

				dct1 = ck * cl * int(iquant_out[k * n + l]) *
					cosf((2 * ((row % 8)) + 1) * (k % 8) * pi_math / (2 * 8)) *
					cosf((2 * (col % 8) + 1) * (l % 8) * pi_math / (2 * 8));
				sum += dct1;
			}
		}
		recon_img[row * n + col] = (unsigned char)sum;
	}
}


// ------------------------------------------------------------------------------------------------
// Below DCT and Inverse DCT Kernels swap the cosf calculations with precomputed values
// Uses Global Memory with high L1 Cache hit rate.
// Reduced Registers from 108 to 40 with use of precomputed memory, allows more threads per block
// ------------------------------------------------------------------------------------------------

/*
	DCT Kernel - Reduced Registers
	input  -> unsigned char* img_in is the input image data
	output -> float* dct_out is the transformed output data

	int m is the number of rows
	int n is the number of cols
	both m and n are default 8 for 8x8 DCT
*/
__global__ void dct_kernel2(unsigned char* img_in, 
	float* dct_out, 
	float* cosk, float* cosl, 
	int m = 8, int n = 8)
{
	// global row and col
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int k, l;
	float ci, cj, dct1, sum;

	// boundary check
	if (row < m && col < n) {
		// theory: if row is row 0 of current block
		if (row % 8 == 0)
			ci = 1 / sqrtf(8);
		else
			ci = sqrtf(2) / sqrtf(8);
		// theory: if col is col 0 of current block
		if (col % 8 == 0)
			cj = 1 / sqrtf(8);
		else
			cj = sqrtf(2) / sqrtf(8);

		sum = 0;

		// theory: start in the top left corner of the 8x8 block
		int startIdx_row = row - (row % 8);
		int startIdx_col = col - (col % 8);
		for (k = startIdx_row; k < startIdx_row + 8; k++) {
			for (l = startIdx_col; l < startIdx_col + 8; l++) {
				dct1 = int(img_in[k * n + l]) * 
					cosk[(row % 8) * 512 + (col % 8) * 64 + (k % 8) * 8 + (l % 8)] *
					cosl[(row % 8) * 512 + (col % 8) * 64 + (k % 8) * 8 + (l % 8)];
				sum += dct1;
			}
		}
		dct_out[row * n + col] = ci * cj * sum;
	}
}


/*
	Inverse DCT Kernel - Reduced Registers
	input  -> int *iquant_out is the inverse quantized results
	output -> unsigned char* recon_img is the inverse dct, reconstructed image
*/
__global__ void idct_kernel2(int* iquant_out, unsigned char* recon_img, float* cosk, float* cosl, int m = 8, int n = 8)
{
	// global row and col
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	int k, l;
	float ck, cl, dct1, sum;

	// boundary check
	if (row < m && col < n) {

		sum = 0;
		// theory: start in the top left corner of the 8x8 block
		int startIdx_row = row - (row % 8);
		int startIdx_col = col - (col % 8);
		for (k = startIdx_row; k < startIdx_row + 8; k++) {
			for (l = startIdx_col; l < startIdx_col + 8; l++) {
				// theory: if row is row 0 of current block
				if (k % 8 == 0)
					ck = 1 / sqrtf(8);
				else
					ck = sqrtf(2) / sqrtf(8);
				// theory: if col is col 0 of current block
				if (l % 8 == 0)
					cl = 1 / sqrtf(8);
				else
					cl = sqrtf(2) / sqrtf(8);

				dct1 = ck * cl * int(iquant_out[k * n + l]) *
					cosk[(k % 8) * 512 + (l % 8) * 64 + (row % 8) * 8 + (col % 8)] *
					cosl[(k % 8) * 512 + (l % 8) * 64 + (row % 8) * 8 + (col % 8)];
				sum += dct1;
			}
		}
		recon_img[row * n + col] = (unsigned char)sum;
	}
}
#include <math.h>
#include <stdio.h>
#include <list>

#define pi_math 3.142857


/* 
	Function to print matrix with elements of type float
	args:
		float *block is the matrix to be printed
		int nrow is the number of rows
		int ncol is the number of cols
		
		both nrow and ncol are default 8
*/
void printBlock_f(float *block, int nrow=8, int ncol=8) 
{
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%f\t", block[i*nrow+j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* 
	Function to print matrix with elements of type int
	args:
		int *block is the matrix to be printed
		int nrow is the number of rows
		int ncol is the number of cols
		
		both nrow and ncol are default 8
*/
void printBlock_i(int *block, int nrow=8, int ncol=8)  {
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%d\t", block[i*nrow+j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* 
	Function to print list with elements of type int
	args:
		std::list<int> l1 is the list to be printed
*/
void printList_i(std::list<int>& l1) {
    std::list<int>::iterator it;
    int i = 0;
    for (it = l1.begin(); it != l1.end(); it++) {
        if ((i+1)%2 == 0)
            printf(" %d) ", *it);
        else
            printf("(%d,", *it);
        i++;
    }
    printf("\n");
}

void printList_c(std::list<char>& l1) {
    std::list<char>::iterator it;
    int i = 0;
    for (it = l1.begin(); it != l1.end(); it++) {
        if ((i + 1) % 2 == 0)
            printf(" %u) ", *it);
        else
            printf("(%u,", *it);
        i++;
    }
    printf("\n");
}

/* 
	Function to print matrix with elements of type unsigned char
	args:
		int *block is the matrix to be printed
		int nrow is the number of rows
		int ncol is the number of cols
		
		both nrow and ncol are default 8
*/
void printBlock_uc(unsigned char *block, int nrow=8, int ncol=8) {
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%u\t", block[i*nrow+j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*
------------------------------------------------------------------------------------------------------------------------------------------------------
ENCODER 

	dct function to transform 8x8 block

	quant function to quantize 8x8 block

	zig zag function to reorganize 8x8 block into 64-d vector

	rle function to generate pairs from 64-d vector

*/

/*
	Function to find discrete cosine transform of 8x8 block
	args:
		int *block is the 8x8 block to be transformed
		float *dct_out is the 8x8 output of the dct function
		int m is the number of rows
		int n is the number of cols
		
		both m and n are default 8 for 8x8 DCT
		
	Adaptation of https://www.geeksforgeeks.org/discrete-cosine-transform-algorithm-program/
*/
void dct(unsigned char *block, float *dct_out, int m=8, int n=8) {
    int i, j, k, l;
 
    float ci, cj, dct1, sum;
 
    //printf("Looping 8x8 block and performing cosine calculations\n");
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
 
            // ci and cj depends on frequency as well as
            // number of row and columns of specified block
            if (i == 0)
                ci = 1 / sqrtf(8);
            else
                ci = sqrtf(2) / sqrtf(8);
            if (j == 0)
                cj = 1 / sqrtf(8);
            else
                cj = sqrtf(2) / sqrtf(8);
 
            //printf("ci, cj = %f, %f\n", ci, cj);

            // sum will temporarily store the sum of cosine signals
            sum = 0;
            for (k = 0; k < 8; k++) {
                for (l = 0; l < 8; l++) {
					// compute each iteration of the nested sum
                    dct1 = int(block[k*m+l]) *
                           cosf((2 * k + 1) * i * pi_math / (2 * 8)) *
                           cosf((2 * l + 1) * j * pi_math / (2 * 8));
                    sum = sum + dct1;
                }
            }
            dct_out[i*m+j] = ci * cj * sum;
        }
    }
}

/*
	Function to calculate quantized 8x8 block
	args:
		float *block is the 8x8 block to be quantized (comes from dct_out)
		int *quant_out is the 8x8 output of the quant function		
*/
void quant(float *block, int *quant_out) {

    int i, j;

    int q[64] = {16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

    for (i=0;i<8;i++) {
        for(j=0; j<8;j++) {
            quant_out[i*8+j] = block[i*8+j] / q[i*8+j]; //implicit cast to int
        }
    }
}
 
/*
	Function to perform zig zag scan on 8x8 block
	args:
		int *block is the 8x8 block to be scanned (comes from quant_out)
		int *zigzag_out is the 64-d output array of scan
		int m is the number of rows
		int n is the number of cols
		
		both m and n are default 8 for 8x8 zig zag scan
	
	Adaptation of https://www.geeksforgeeks.org/print-matrix-in-zig-zag-fashion/
*/
void zigzag(int *block, int *zigzag_out, int m=8, int n=8) {

    // initialization place
    int row = 0;
    int col = 0;

    // fill out array index
    int idx_out = 0;

    // flag for incremening r or c in iterations of reading scan
    bool row_inc = false;

    // nested loop to compute lower half of zig-zag pattern
    int mn = fmin(m, n);
    for (int len = 1; len <= mn; ++len) {
        for (int i = 0; i < len; ++i) {
            // printf("%d ", block[row*m+col]); // debug the printing
            zigzag_out[idx_out] = block[row*m+col];
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
 
    // Update the indexes of row and col variable
    if (row == 0) {
        if (col == m - 1)
            ++row;
        else
            ++col;
        row_inc = 1;
    }
    else {
        if (row == n - 1)
            ++col;
        else
            ++row;
        row_inc = 0;
    }
 
    // Print the next half zig-zag pattern
    int MAX = fmax(m, n) - 1;
    for (int len, diag = MAX; diag > 0; --diag) {
 
        if (diag > mn)
            len = mn;
        else
            len = diag;
 
        for (int i = 0; i < len; ++i) {
            // printf("%d ", block[row*m+col]); // debug the printing
            zigzag_out[idx_out] = block[row*m+col];
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
 
        // Update the indexes of row and col variable
        if (row == 0 || col == m - 1) {
            if (col == m - 1)
                ++row;
            else
                ++col;
 
            row_inc = true;
        }
 
        else if (col == 0 || row == n - 1) {
            if (row == n - 1)
                ++col;
            else
                ++row;
 
            row_inc = false;
        }
    }
}

/*
	Function to perform run length encoding  (rle) on 64-d array
	args:
		int *zigzag_out is the 64-d array to be encoded (comes from zigzag_out)
		int *rle_out is a list
		
	Adaptation of https://www.geeksforgeeks.org/run-length-encoding/
*/
void rle(int *zigzag_out, std::list<int>& rle_out) {

    for (int i = 0; i < 64; i++) {
 
        // Count occurrences of current character
        int count = 1;
        while (i < 64 - 1 && zigzag_out[i] == zigzag_out[i + 1]) {
            count++;
            i++;
        }
        // Print character and its count
        // printf("%d(%d), ", zigzag_out[i], count);
        rle_out.push_back(zigzag_out[i]);
        rle_out.push_back(count);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------

/*
------------------------------------------------------------------------------------------------------------------------------------------------------
DECODER 

idct function to inverse transform 8x8 block

iquant function to inverse quantize 8x8 block

izigzag function to reorganize 64-d vector into 8x8 block

irle function to generate 64-d vector from rle pairs

*/

/*
	Function to find inverse discrete cosine transform of 8x8 block
	args:
		int *block is the 8x8 block to be transformed (comes from iquant_out)
		float *idct_out is the 8x8 output of the idct function
		int m is the number of rows
		int n is the number of cols
		
		both m and n are default 8 for 8x8 iDCT
*/
void idct(int *block, unsigned char *idct_out, int m=8, int n=8)
{
    int i, j, k, l;
 
    float ck, cl, idct1, sum;
 
    //printf("Looping 8x8 block and performing cosine calculations\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
 
            // sum will temporarily store the sum of cosine signals
            sum = 0;
            for (k = 0; k < m; k++) {
                for (l = 0; l < n; l++) {
					
					// ci and cj depends on frequency as well as
					// number of row and columns of specified block
					if (k == 0)
						ck = 1 / sqrtf(m);
					else
						ck = sqrtf(2) / sqrtf(m);
					if (l == 0)
						cl = 1 / sqrt(n);
					else
						cl = sqrtf(2) / sqrtf(n);
					
                    idct1 = ck * cl * int(block[k*m+l]) *
                           cosf((2 * i + 1) * k * pi_math / (2 * m)) *
                           cosf((2 * j + 1) * l * pi_math / (2 * n));
                    sum = sum + idct1;
                }
            }
            if (sum < 0.0) sum = 0;
            idct_out[i*m+j] = (unsigned char)sum;
            //if (idct_out[i * m + j] > 245) printf("sum > 250: %u\n", idct_out[i * m + j]);
        }
    }
}

/*
	Function to calculate inverse quantized 8x8 block
	args:
		float *block is the 8x8 block to be quantized (comes from izigzag_out)
		int *iquant_out is the 8x8 output of the iquant function
*/
void iquant(int *block, int *iquant_out) {

	// iterators declaration
    int i, j;

	// q matrix
    int q[64] = {16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};

	// iterate the 8x8 block and multiply element wise the block[idx]*q[idx]
    for (i=0;i<8;i++)
        for(j=0; j<8;j++)
            iquant_out[i*8+j] = block[i*8+j] * q[i*8+j]; // implicit cast to int
}

/*
	Function to perform inverse zig zag scan on 8x8 block
	args:
		int *zigzag_out is the 64-d array that holds all the needed values (comes from irle_out)
		int *reconstructed_block is the 8x8 block result of izigzag
		int m is the number of rows
		int n is the number of cols
		
		both m and n are default 8 for 8x8 inverse zig zag scan
		
*/
void izigzag(int *izigzag_out, int *reconstructed_block, int m=8, int n=8) {

    // initialization place
    int row = 0;
    int col = 0;

    // fill out array index
    int idx_out = 0;

    // flag for incremening r or c in iterations of reading scan
    bool row_inc = false;

    // nested loop to compute lower half of zig-zag pattern
    int mn = fmin(m, n);
    for (int len = 1; len <= mn; ++len) {
        for (int i = 0; i < len; ++i) {
            //printf("%d ", zigzag_out[idx_out]); // debug the printing
            reconstructed_block[row*m+col] = izigzag_out[idx_out];
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
 
    // Update the indexes of row and col variable
    if (row == 0) {
        if (col == m - 1)
            ++row;
        else
            ++col;
        row_inc = 1;
    }
    else {
        if (row == n - 1)
            ++col;
        else
            ++row;
        row_inc = 0;
    }
 
    // Print the next half zig-zag pattern
    int MAX = fmax(m, n) - 1;
    for (int len, diag = MAX; diag > 0; --diag) {
 
        if (diag > mn)
            len = mn;
        else
            len = diag;
 
        for (int i = 0; i < len; ++i) {
            // printf("%d ", block[row*m+col]); // debug the printing
            reconstructed_block[row*m+col] = izigzag_out[idx_out];
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
 
        // Update the indexes of row and col variable
        if (row == 0 || col == m - 1) {
            if (col == m - 1)
                ++row;
            else
                ++col;
 
            row_inc = true;
        }
 
        else if (col == 0 || row == n - 1) {
            if (row == n - 1)
                ++col;
            else
                ++row;
 
            row_inc = false;
        }
    }
}

void irle(std::list<int>& rle_compressed, int* zigzag_out) {

    // reconstruct the zig zag out based on rle compressed representation
    std::list<int>::iterator it;
    int i = 0;
    int tracker_idx = 0;
    int v = 0;
    int rep = 0;
    for (it = rle_compressed.begin(); it != rle_compressed.end(); it++) {
        if ((i+1)%2 == 0) {
            rep =  *it;
            for (int j = 0; j < rep; j++) {
                zigzag_out[tracker_idx+j] = v;
            }
            tracker_idx += rep;
        }
        else
            v = *it;
        i++;
    }
}
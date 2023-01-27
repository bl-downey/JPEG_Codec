#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <vector>

// serial codec
#include "../utils/codec_serial.h"
//parallel codec
#include "../utils/codec_parallel.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../utils/stb_image_write.h"

// #include your error-check macro header file
#include "../utils/CUDA_error_check.h"

// timers
#include "C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um\profileapi.h"

/*
Function to run the JPEG Codec on the input image, img, on GPU Kernels
*/
void jpeg_codec_parallel(unsigned char * img, std::vector<std::list<char>> encoded, unsigned char* imgDataOut, int m, int n, bool dbg=false) {

    int imgSize = m * n * sizeof(unsigned char); // size of memory for uchar image
    int imgSize_f = m * n * sizeof(float);	     // size of memory for float image transform
    int imgSize_i = m * n * sizeof(int);		 // size of memory for all other steps - int
    int qSize = BLOCK_SIZE * sizeof(int);		 // size of q matrix

    // allocate memory for host variables
    int* h_encoded_val = (int*)malloc(imgSize_i);
    int* h_encoded_rep = (int*)malloc(imgSize_i);

    int q[64] = { 16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };

    // allocate device memory
    unsigned char* dev_in = 0;	 // input image
    float* dct_out = 0; 		 // output of dct since dct output is float
    int* quant_out = 0; 		 // output of quant
    int* zigzag_out = 0;         // output of zigzag
    int* encoded_val = 0;        // output of rle - values
    int* encoded_rep = 0;        // output of rle - length
    int* irle_out = 0;           // output of irle
    int* dev_q = 0;				 // q matrix

    // Init GPU - 0
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers
    checkCuda(cudaMalloc((void**)&dev_in, imgSize));		 // input image - unsigned char
    checkCuda(cudaMalloc((void**)&dct_out, imgSize_f));		 // dct out transform - float
    checkCuda(cudaMalloc((void**)&quant_out, imgSize_i));	 // quant(iquant) out - int
    checkCuda(cudaMalloc((void**)&zigzag_out, imgSize_i));	 // zigzag(izigzag) out - int
    checkCuda(cudaMalloc((void**)&encoded_val, imgSize_i));	 // encoded values - int
    checkCuda(cudaMalloc((void**)&encoded_rep, imgSize_i));	 // encoded lengths - int
    checkCuda(cudaMalloc((void**)&irle_out, imgSize_i));	 // inverse rle out - int (can use quant_out for this but didn't)
    checkCuda(cudaMalloc((void**)&dev_q, qSize));			 // q matrix

    // copy host data to device
    checkCuda(cudaMemcpy(dev_in, img, imgSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_q, q, qSize, cudaMemcpyHostToDevice));

    // Kernel Launch Configurations
    // These used for DCT, Quant and iDCT, iQuant - 1 thread per value in matrix
    
    //512x512 image
    dim3 DimBlock_dct(16, 16, 1); // 256 threads
    dim3 DimGrid_dct(32, 32, 1);  // 1024 blocks

    if (m == 240) { // 240x240 image
        DimBlock_dct.x = 16; // 256 threads
        DimBlock_dct.y = 16;
        DimGrid_dct.x = 15;  // 225 blocks
        DimGrid_dct.y = 15;
    }

    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_dct.x, DimGrid_dct.y, DimGrid_dct.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_dct.x, DimBlock_dct.y, DimBlock_dct.z);
    }

    // ***************** DCT KERNEL *****************
    dct_kernel1 << <DimGrid_dct, DimBlock_dct >> > (dev_in, dct_out, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** QUANT KERNEL *****************
    quant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (dct_out, quant_out, dev_q, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** ZIG ZAG KERNEL *****************
    // zig zag kernel uses different kernel launch configs
    int b = m * n / 64;         // number of 8x8 blocks
    
    dim3 DimBlock_zz(32, 32, 1); //512x512 image
    dim3 DimGrid_zz(2, 2, 1); 

    if (m == 240) { // 240x240 image
        DimBlock_zz.x = 10;
        DimBlock_zz.y = 10;
        DimGrid_zz.x = 3;
        DimGrid_zz.y = 3;
    }
    
    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_zz.x, DimGrid_zz.y, DimGrid_zz.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_zz.x, DimBlock_zz.y, DimBlock_zz.z);
    }
    // launch zig zag kernel
    zigzag_kernel1 << <DimGrid_zz, DimBlock_zz >> > (quant_out, zigzag_out, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** RUN LENGTH ENCODING KERNEL *****************
    rle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (zigzag_out, encoded_val, encoded_rep, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    checkCuda(cudaMemcpy(h_encoded_val, encoded_val, imgSize_i, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_encoded_rep, encoded_rep, imgSize_i, cudaMemcpyDeviceToHost));

    // GENERATE ENCODED REPRESENTATION
    int bytes = 0;
    for (std::size_t i = 0; i < encoded.size(); i++) {
        std::list<char> block_compression;
        for (int j = 0; j < 64; j++) {
            if (h_encoded_rep[i * 64 + j] == 0) break; // remove all remaining pairs that take up excess memory
            block_compression.push_back((char)(h_encoded_val[i * 64 + j]));
            block_compression.push_back((char)(h_encoded_rep[i * 64 + j]));
            bytes += 2;
        }
        encoded[i] = block_compression;
    }
    if (dbg) {
        printf("Encoded Representation:\n");
        for (int i = 0; i < b; i++)
            printList_c(encoded[i]);
        printf("\n");
    }

    printf("Compression from %d to %d bytes\n", imgSize, bytes* sizeof(char));

    // DECODE ENCODED REPRESENTATION
    std::list<char>::iterator it;
    int rep_idx = 0;
    int val_idx = 0;
    int recon_idx = 0;

    for (int i = 0; i < b; i++) {
        // append the list elements for the encoded[i] list block to the encoded arrays  
        for (it = encoded[i].begin(); it != encoded[i].end(); it++) {
            if (((recon_idx + 1) % 2) == 0) {
                h_encoded_rep[rep_idx] = *it; rep_idx++;
            }
            else {
                h_encoded_val[val_idx] = *it; val_idx++;
            }
            recon_idx++;
        }
        for (int j = rep_idx; j < ((i + 1) * 64); j++) {
            h_encoded_rep[j] = 0;
            h_encoded_val[j] = 0;
        }
        rep_idx = (i + 1) * 64;
        val_idx = (i + 1) * 64;
    }

    checkCuda(cudaMemcpy(encoded_val, h_encoded_val, imgSize_i, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(encoded_rep, h_encoded_rep, imgSize_i, cudaMemcpyHostToDevice));

    // ***************** INVERSE RUN LENGTH ENCODING KERNEL *****************
    irle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (encoded_val, encoded_rep, irle_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE ZIG ZAG KERNEL *****************
    izigzag_kernel1 << <DimGrid_zz, DimBlock_zz >> > (irle_out, zigzag_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE QUANT KERNEL *****************
    iquant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (zigzag_out, quant_out, dev_q, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE DCT KERNEL *****************
    idct_kernel1 << <DimGrid_dct, DimBlock_dct >> > (quant_out, dev_in, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // transfer image data back to host 
    checkCuda(cudaMemcpy(imgDataOut, dev_in, imgSize, cudaMemcpyDeviceToHost));

    checkCuda(cudaDeviceReset());
}

/*
Function to run the JPEG Codec on the input image, img, on GPU Kernels
Uses different dct and idct kernel to test reduced register, higher thread / block tradeoff
*/
void jpeg_codec_parallel_rr(unsigned char* img, 
    std::vector<std::list<char>> encoded,
    unsigned char* imgDataOut,
    float* cosk, float* cosl,
    int m, int n,
    bool dbg = false) {

    int imgSize = m * n * sizeof(unsigned char); // size of memory for uchar image
    int imgSize_f = m * n * sizeof(float);	     // size of memory for float image transform
    int imgSize_i = m * n * sizeof(int);		 // size of memory for all other steps - int
    int qSize = BLOCK_SIZE * sizeof(int);		 // size of q matrix

    // allocate memory for host variables
    int* h_encoded_val = (int*)malloc(imgSize_i);
    int* h_encoded_rep = (int*)malloc(imgSize_i);

    int q[64] = { 16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };

    // allocate device memory
    unsigned char* dev_in = 0;	 // input image
    float* dct_out = 0; 		 // output of dct since dct output is float
    int* quant_out = 0; 		 // output of quant
    int* zigzag_out = 0;         // output of zigzag
    int* encoded_val = 0;        // output of rle - values
    int* encoded_rep = 0;        // output of rle - length
    int* irle_out = 0;           // output of irle
    int* dev_q = 0;				 // q matrix
    float* cosk_dct = 0;		 // cosk dct
    float* cosl_dct = 0;		 // cosl dct

    // Init GPU - 0
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers
    checkCuda(cudaMalloc((void**)&dev_in, imgSize));		 // input image - unsigned char
    checkCuda(cudaMalloc((void**)&dct_out, imgSize_f));		 // dct out transform - float
    checkCuda(cudaMalloc((void**)&quant_out, imgSize_i));	 // quant(iquant) out - int
    checkCuda(cudaMalloc((void**)&zigzag_out, imgSize_i));	 // zigzag(izigzag) out - int
    checkCuda(cudaMalloc((void**)&encoded_val, imgSize_i));	 // encoded values - int
    checkCuda(cudaMalloc((void**)&encoded_rep, imgSize_i));	 // encoded lengths - int
    checkCuda(cudaMalloc((void**)&irle_out, imgSize_i));	 // inverse rle out - int (can use quant_out for this but didn't)
    checkCuda(cudaMalloc((void**)&dev_q, qSize));			 // q matrix
    checkCuda(cudaMalloc((void**)&cosk_dct, 4096 * sizeof(float))); //cosk GMEM for dct
    checkCuda(cudaMalloc((void**)&cosl_dct, 4096 * sizeof(float))); //cosl GMEM for dct

    // copy host data to device
    checkCuda(cudaMemcpy(dev_in, img, imgSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_q, q, qSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(cosk_dct, cosk, 4096 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(cosl_dct, cosl, 4096 * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel Launch Configurations
    // These used for DCT, Quant and iDCT, iQuant - 1 thread per value in matrix

    //512x512 image
    dim3 DimBlock_dct(32, 32, 1); // 1024 threads
    dim3 DimGrid_dct(16, 16, 1);  // 256 blocks

    if (m == 240) { // 240x240 image
        DimBlock_dct.x = 32; // 1024 threads
        DimBlock_dct.y = 32;
        DimGrid_dct.x = 8;  // 225 blocks
        DimGrid_dct.y = 8;
    }

    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_dct.x, DimGrid_dct.y, DimGrid_dct.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_dct.x, DimBlock_dct.y, DimBlock_dct.z);
    }

    // ***************** DCT KERNEL *****************
    dct_kernel2 << <DimGrid_dct, DimBlock_dct >> > (dev_in, dct_out, cosk_dct, cosl_dct, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** QUANT KERNEL *****************
    quant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (dct_out, quant_out, dev_q, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** ZIG ZAG KERNEL *****************
    // zig zag kernel uses different kernel launch configs
    int b = m * n / 64;         // number of 8x8 blocks

    dim3 DimBlock_zz(32, 32, 1); //512x512 image
    dim3 DimGrid_zz(2, 2, 1);

    if (m == 240) { // 240x240 image
        DimBlock_zz.x = 10;
        DimBlock_zz.y = 10;
        DimGrid_zz.x = 3;
        DimGrid_zz.y = 3;
    }

    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_zz.x, DimGrid_zz.y, DimGrid_zz.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_zz.x, DimBlock_zz.y, DimBlock_zz.z);
    }
    // launch zig zag kernel
    zigzag_kernel1 << <DimGrid_zz, DimBlock_zz >> > (quant_out, zigzag_out, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** RUN LENGTH ENCODING KERNEL *****************
    rle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (zigzag_out, encoded_val, encoded_rep, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    checkCuda(cudaMemcpy(h_encoded_val, encoded_val, imgSize_i, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_encoded_rep, encoded_rep, imgSize_i, cudaMemcpyDeviceToHost));

    // GENERATE ENCODED REPRESENTATION
    int bytes = 0;
    for (std::size_t i = 0; i < encoded.size(); i++) {
        std::list<char> block_compression;
        for (int j = 0; j < 64; j++) {
            if (h_encoded_rep[i * 64 + j] == 0) break; // remove all remaining pairs that take up excess memory
            block_compression.push_back((char)(h_encoded_val[i * 64 + j]));
            block_compression.push_back((char)(h_encoded_rep[i * 64 + j]));
            bytes += 2;
        }
        encoded[i] = block_compression;
    }
    if (dbg) {
        printf("Encoded Representation:\n");
        for (int i = 0; i < b; i++)
            printList_c(encoded[i]);
        printf("\n");
    }

    printf("Compression from %d to %d bytes\n", imgSize, bytes * sizeof(char));

    // DECODE ENCODED REPRESENTATION
    std::list<char>::iterator it;
    int rep_idx = 0;
    int val_idx = 0;
    int recon_idx = 0;

    for (int i = 0; i < b; i++) {
        // append the list elements for the encoded[i] list block to the encoded arrays  
        for (it = encoded[i].begin(); it != encoded[i].end(); it++) {
            if (((recon_idx + 1) % 2) == 0) {
                h_encoded_rep[rep_idx] = *it; rep_idx++;
            }
            else {
                h_encoded_val[val_idx] = *it; val_idx++;
            }
            recon_idx++;
        }
        for (int j = rep_idx; j < ((i + 1) * 64); j++) {
            h_encoded_rep[j] = 0;
            h_encoded_val[j] = 0;
        }
        rep_idx = (i + 1) * 64;
        val_idx = (i + 1) * 64;
    }

    checkCuda(cudaMemcpy(encoded_val, h_encoded_val, imgSize_i, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(encoded_rep, h_encoded_rep, imgSize_i, cudaMemcpyHostToDevice));

    // ***************** INVERSE RUN LENGTH ENCODING KERNEL *****************
    irle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (encoded_val, encoded_rep, irle_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE ZIG ZAG KERNEL *****************
    izigzag_kernel1 << <DimGrid_zz, DimBlock_zz >> > (irle_out, zigzag_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE QUANT KERNEL *****************
    iquant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (zigzag_out, quant_out, dev_q, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    // ***************** INVERSE DCT KERNEL *****************
    idct_kernel2 << <DimGrid_dct, DimBlock_dct >> > (quant_out, dev_in, cosk_dct, cosl_dct, m, n);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // transfer image data back to host 
    checkCuda(cudaMemcpy(imgDataOut, dev_in, imgSize, cudaMemcpyDeviceToHost));

    checkCuda(cudaDeviceReset());
}

/*
Function to run a multiblock matrix through the Compression and Decompression stages on GPU Kernel
*/
void gpu_unit_test(float* cosk, float* cosl, bool dbg = false) {
    const int m = 16;
    const int n = 16;
    unsigned char img[m * n];
    unsigned char img_out[m * n];
    std::list<int> encoded[m * n / 64];

    printf("Running GPU Unit Test on sample %dx%d block:\n", m, n);

    if (dbg)
        printf("Filling sample %dx%d img:\n", m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            img[i * n + j] = (unsigned char)255; //i * m + j | 255 | rand() % 256

    if (dbg) {
        printf("Sample %dx%d img:\n", m, n);
        printBlock_uc(img, m, n);
    }

    int imgSize = m * n * sizeof(unsigned char); // size of memory for uchar image
    int imgSize_f = m * n * sizeof(float);	     // size of memory for float image transform
    int imgSize_i = m * n * sizeof(int);		 // size of memory for all other steps - int
    int qSize = BLOCK_SIZE * sizeof(int);		 // size of q matrix

    // allocate memory for host variables
    if (dbg)
        printf("Allocate Host Memory...\n");
    float* h_dct_out = (float*)malloc(imgSize_f);
    int* h_quant_out = (int*)malloc(imgSize_i);
    int* h_iquant_out = (int*)malloc(imgSize_i);
    int* h_zigzag_out = (int*)malloc(imgSize_i);
    int* h_izigzag_out = (int*)malloc(imgSize_i);
    int* h_encoded_val = (int*)malloc(imgSize_i);
    int* h_encoded_rep = (int*)malloc(imgSize_i);
    int* h_iencoded_val = (int*)malloc(imgSize_i);
    int* h_iencoded_rep = (int*)malloc(imgSize_i);
    int* h_irle_out = (int*)malloc(imgSize_i);

    int q[64] = { 16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };

    // allocate device memory
    if (dbg)
        printf("Allocate Device Memory...\n");
    unsigned char* dev_in = 0;	 // input image
    unsigned char* dev_out = 0;	 // output image
    float* dct_out = 0; 		 // output of dct since dct output is float
    int* quant_out = 0; 			 // output of quant and zigzag since these are
    int* iquant_out = 0; 			 // output of quant and zigzag since these are
    int* zigzag_out = 0;
    int* izigzag_out = 0;
    int* encoded_val = 0;
    int* encoded_rep = 0;
    int* iencoded_val = 0;
    int* iencoded_rep = 0;
    int* irle_out = 0;
    int* dev_q = 0;				 // q matrix
    float* cosk_dct = 0;				 // cosk dct
    float* cosl_dct = 0;				 // cosl dct

    // Init GPU - 0
    checkCuda(cudaSetDevice(0));

    // Allocate GPU buffers
    if (dbg)
        printf("Allocate GPU Buffers...\n");
    checkCuda(cudaMalloc((void**)&dev_in, imgSize));		// input image
    checkCuda(cudaMalloc((void**)&dev_out, imgSize));		// input image
    checkCuda(cudaMalloc((void**)&dct_out, imgSize_f));		// dct out transform
    checkCuda(cudaMalloc((void**)&quant_out, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&iquant_out, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&zigzag_out, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&izigzag_out, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&encoded_val, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&encoded_rep, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&iencoded_val, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&iencoded_rep, imgSize_i));	// int type temp data
    checkCuda(cudaMalloc((void**)&irle_out, imgSize_i));	// int type temp data
    // q matrix allocation
    checkCuda(cudaMalloc((void**)&dev_q, qSize));			// q matrix
    checkCuda(cudaMalloc((void**)&cosk_dct, 4096 * sizeof(float))); //cosk GMEM for dct
    checkCuda(cudaMalloc((void**)&cosl_dct, 4096 * sizeof(float))); //cosl GMEM for dct

    // copy host data to device
    if (dbg)
        printf("Copy Host Data to GPU Buffers...\n");
    checkCuda(cudaMemcpy(dev_in, img, imgSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_q, q, qSize, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(cosk_dct, cosk, 4096 * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(cosl_dct, cosl, 4096 * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel Launch Configuration
    dim3 DimBlock_dct(8, 8, 1);
    dim3 DimGrid_dct((m - 1) / 8 + 1, (n - 1) / 8 + 1, 1);
    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_dct.x, DimGrid_dct.y, DimGrid_dct.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_dct.x, DimBlock_dct.y, DimBlock_dct.z);
    }

    // ***************** TEST DCT KERNEL *****************
    if (dbg)
        printf("Launching DCT Kernel\n");
    dct_kernel2 << <DimGrid_dct, DimBlock_dct >> > (dev_in, dct_out, cosk_dct, cosl_dct, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(h_dct_out, dct_out, imgSize_f, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("DCT Out Results:\n");
        printBlock_f(h_dct_out, m, n);
    }

    // ***************** TEST QUANT KERNEL *****************
    if (dbg)
        printf("Launching Quantization Kernel\n");
    quant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (dct_out, quant_out, dev_q, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(h_quant_out, quant_out, imgSize_i, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Quant Out Results:\n");
        printBlock_i(h_quant_out, m, n);
    }
    
    // ***************** TEST ZIG ZAG KERNEL *****************
    // zig zag kernel uses different kernel launch configs
    int b = m * n / 64;         // number of 8x8 blocks

    dim3 DimBlock_zz(2, 2, 1);
    dim3 DimGrid_zz(1, 1, 1); // square grid
    if (dbg) {
        printf("DimGrid:  (%d, %d, %d)\n", DimGrid_zz.x, DimGrid_zz.y, DimGrid_zz.z);
        printf("DimBlock: (%d, %d, %d)\n", DimBlock_zz.x, DimBlock_zz.y, DimBlock_zz.z);
    }
    // launch zig zag kernel
    if (dbg)
        printf("Launching Zig Zag Kernel\n");
    zigzag_kernel1<<<DimGrid_zz, DimBlock_zz>>> (quant_out, zigzag_out, m ,n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
	checkCuda(cudaMemcpy(h_izigzag_out, zigzag_out, imgSize_i, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Print Zig Zag Scan Result:\n");
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < 64; j++) {
                printf("%d ", h_izigzag_out[i * 64 + j]);
            }
            printf("\n");
        }
    }
    printf("\n");

    // ***************** TEST RUN LENGTH ENCODING KERNEL *****************
    // launch rle kernel
    if (dbg)
        printf("Launching Run Length Encoding Kernel\n");
    rle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (zigzag_out, encoded_val, encoded_rep, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(h_encoded_val, encoded_val, imgSize_i, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_encoded_rep, encoded_rep, imgSize_i, cudaMemcpyDeviceToHost));

    // print results
    if (dbg) {
        printf("RLE Results:\n");
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < 64; j++) {
                printf("%d(%d) ", h_encoded_val[i*64+j], h_encoded_rep[i * 64 + j]);
            }
            printf("\n\n");
        }
    }

    // GENERATE ENCODED REPRESENTATION
    printf("RLE Results:\n");
    for (int i = 0; i < b; i++) {
        std::list<int> block_compression;
        for (int j = 0; j < 64; j++) {
            if (h_encoded_rep[i * 64 + j] == 0) break; // remove all remaining pairs that take up excess memory
            block_compression.push_back(h_encoded_val[i * 64 + j]);
            block_compression.push_back(h_encoded_rep[i * 64 + j]);
        }
        encoded[i] = block_compression;
    }
    if (dbg) {
        printf("Encoded Representation:\n");
        for (int i = 0; i < b; i++)
            printList_i(encoded[i]);
        printf("\n");
    }

    // DECODE ENCODED REPRESENTATION
    std::list<int>::iterator it;
    int rep_idx = 0;
    int val_idx = 0;
    int recon_idx = 0;

    for (int i = 0; i < b; i++) {
        // append the list elements for the encoded[i] list block to the encoded arrays  
        for (it = encoded[i].begin(); it != encoded[i].end(); it++) {
            if (((recon_idx + 1) % 2) == 0) {
                h_iencoded_rep[rep_idx] = *it; rep_idx++;
            }
            else {
                h_iencoded_val[val_idx] = *it; val_idx++;
            }
            recon_idx++;
        }
        for (int j = rep_idx; j < ((i+1)*64); j++) {
            h_iencoded_rep[j] = 0;
            h_iencoded_val[j] = 0;
        }
        rep_idx = (i + 1) * 64;
        val_idx = (i + 1) * 64;
    }

    // print results
    if (dbg) {
        printf("Decoded Representation:\n");
        for (int i = 0; i < b; i++) {
            printf("Block %d:\n", i);
            for (int j = 0; j < 64; j++)
                printf("%d(%d) ", h_iencoded_val[i * 64 + j], h_iencoded_rep[i * 64 + j]);
            printf("\n\n");
        }
    }

    checkCuda(cudaMemcpy(iencoded_val, h_iencoded_val, imgSize_i, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(iencoded_rep, h_iencoded_rep, imgSize_i, cudaMemcpyHostToDevice));

    /*int* h_q = (int*)malloc(qSize);
    checkCuda(cudaMemcpy(h_q, dev_q, qSize, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 64; i++)
        printf("%d ", h_q[i]);
    printf("\n");*/

    // ***************** TEST INVERSE RUN LENGTH ENCODING KERNEL *****************
    // launch rle kernel
    if (dbg)
        printf("Launching Inverse Run Length Encoding Kernel\n");
    irle_kernel1 << <DimGrid_zz, DimBlock_zz >> > (iencoded_val, iencoded_rep, irle_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(h_irle_out, irle_out, imgSize_i, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Print Inverse RLE Result:\n");
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < 64; j++) {
                printf("%d ", h_irle_out[i * 64 + j]);
            }
            printf("\n");
        }
    }
    printf("\n");

    // ***************** TEST INVERSE ZIG ZAG KERNEL *****************
    // launch inverse zig zag kernel
    if (dbg)
        printf("Launching Inverse Zig Zag Kernel\n");
    izigzag_kernel1 << <DimGrid_zz, DimBlock_zz >> > (irle_out, izigzag_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(h_izigzag_out, izigzag_out, imgSize_i, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Inverse Zig Zag Out Results:\n");
        printBlock_i(h_izigzag_out, m, n);
    }

    // ***************** TEST INVERSE QUANT KERNEL *****************
    checkCuda(cudaMemcpy(dev_q, q, qSize, cudaMemcpyHostToDevice));
    if (dbg)
        printf("Launching Inverse Quantization Kernel\n");
    iquant_kernel1 << <DimGrid_dct, DimBlock_dct >> > (izigzag_out, iquant_out, dev_q, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    checkCuda(cudaMemcpy(h_iquant_out, iquant_out, imgSize_i, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Inverse Quant Out Results:\n");
        printBlock_i(h_iquant_out, m, n);
    }

    // ***************** TEST INVERSE DCT KERNEL *****************
    if (dbg)
        printf("Launching Inverse DCT Kernel\n");
    idct_kernel1 << <DimGrid_dct, DimBlock_dct >> > (iquant_out, dev_out, m, n);
    // Check for any errors launching the kernel + device synch to wait to complete
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    // retrieve result data from device back to host
    if (dbg)
        printf("Copy Memory Back to Host\n");
    checkCuda(cudaMemcpy(img_out, dev_out, imgSize, cudaMemcpyDeviceToHost));
    // print results
    if (dbg) {
        printf("Inverse DCT Out Results:\n");
        printBlock_uc(img_out, m, n);
    }

    checkCuda(cudaDeviceReset());
}

/*
Function to run a multiblock matrix through the Compression and Decompression stages
*/
void cpu_unit_test() {
    // declare the matrix and fill it with 256 for test
    const int m = 8;
    const int n = 8;
    unsigned char img[m*n];

    printf("Running CPU Unit Test on sample %dx%d block:\n", m, n);

    printf("Filling sample %dx%d img:\n", m,n);
    for (int i=0;i<m;i++) {
        for (int j=0;j<m;j++)
            img[i*m+j] = (unsigned char)255; //255; //rand() % 256; 
    }
    printf("Sample %dx%d img:\n", m,n);
    printBlock_uc(img, m, n);

    int inc_row = 0;

    // phased (p) algorithm
    for (int p=0; p < int((m*n)/64); p++) {
        // indicates a new row of blocks
		if(p==((n/8)*(inc_row+1))) {
            inc_row++;
        }
        // load phase indices
        //printf("phase %d\n", p);
        unsigned char block[8*8];
        float block_dct_out[8*8];
		unsigned char iblock_dct_out[8*8];
        int quant_out[8*8];
        int zigzag_out[8*8];
		int izigzag_out[8*8];
		int iquant_out[8*8];
        std::list<int> block_list;
        int i_b = 0; int j_b = 0;        
        for (int i=(inc_row*7); i<(inc_row*7)+8; i++) {
            j_b=0;
            for (int j=(p*8); j<(p*8)+8; j++) {
                //printf("%d ", img[i*m+j]);
                block[i_b*8+j_b] = img[i*m+j]; //fill the block
				
				// initialize extra memory for steps to print values to compare
                block_dct_out[i_b*8+j_b] = 0.0;
				iblock_dct_out[i_b*8+j_b] = (unsigned  char) 0;
                quant_out[i_b*8+j_b] = 0.0;
				iquant_out[i_b*8+j_b] = 0.0;
                zigzag_out[i_b*8+j_b] = 0.0;
				izigzag_out[i_b*8+j_b] = 0.0;
                j_b++;
            }
            i_b++; 
            //printf("\n");
        }
        printf("\n");
        
        // print phase number and the block to be processed
        printf("Phase %d - 8x8 block:\n",p);
        printBlock_uc(block);
        // dct on the block
        dct(block, block_dct_out, 8, 8);
        printf("DCT results of 8x8 block:\n");
        printBlock_f(block_dct_out);
        // quant on the block
        quant(block_dct_out, quant_out);
        printf("Quant results of 8x8 block:\n");
        printBlock_i(quant_out);
        // zig zag on the block -> 64-d array
        zigzag(quant_out, zigzag_out, 8, 8);
        printf("Print 64-d Zig Zag Scan Result:\n");
        for(int i=0;i<8*8;i++) {
            printf("%d ", zigzag_out[i]);
        }
        printf("\n\n");

        // RLE on 64-d array
        rle(zigzag_out, block_list); //, block_list
        printf("Run Length Encoding Results of 64-d array:\n");
        printList_i(block_list);
		
		// RLE -> zigzag out
        irle(block_list, zigzag_out);
        printf("Inverse RLE Result:\n");
        for(int i=0;i<8*8;i++) {
            printf("%d ", zigzag_out[i]);
        }
        printf("\n\n");

		// inverse zig zag -> 8x8 block
		izigzag(zigzag_out, izigzag_out);
		printf("Output of Inverse ZigZag\n");
		printBlock_i(izigzag_out);
		printf("\n");
		// inverse quantization on 8x8 block
		iquant(izigzag_out, iquant_out);
		printf("Inverse Quant results of 8x8 block:\n");
		printBlock_i(iquant_out);
		// inverse dct on 8x8 block
		idct(iquant_out, iblock_dct_out);
		printf("Inverse DCT results of 8x8 block:\n");
		printBlock_uc(iblock_dct_out);

        printf("--------------------------------------------------------------------\n");
    }
}

/*
Function to run the JPEG Codec on the input image, img
*/
void jpeg_codec_sequential(unsigned char * img, unsigned char * imgDataOut, int m, int n, bool dbg=false)  {
	
	int inc_row = 0;
	
	// phased (p) algorithm
    for (int p=0; p < int((m*n)/64); p++) {
        // indicates a new row of blocks
		if(p==((n/8)*(inc_row+1))) {
            inc_row++;
        }
        // load phase indices
        unsigned char block[8*8];
        float block_dct_out[8*8];
		unsigned char iblock_dct_out[8*8];
        int quant_out[8*8];
		int iquant_out[8*8];
        int zigzag_out[8*8];
		int izigzag_out[8*8];
        std::list<int> block_list;
		
        int i_b = 0; int j_b = 0;        
        for (int i=(inc_row*7); i<(inc_row*7)+8; i++) {
            j_b=0;
            for (int j=(p*8); j<(p*8)+8; j++) {
				
                block[i_b*8+j_b] = img[i*m+j]; //fill the block
				
				// initialize extra memory blocks to 0.0
                block_dct_out[i_b*8+j_b] = 0.0;
				iblock_dct_out[i_b*8+j_b] = (unsigned char) 0;
                quant_out[i_b*8+j_b] = 0;
				iquant_out[i_b*8+j_b] = 0;
                zigzag_out[i_b*8+j_b] = 0;
				izigzag_out[i_b*8+j_b] = 0;
                j_b++;
            }
            i_b++; 
        }
        //printf("\n");
        
        // print phase number and the block to be processed
        if (dbg) {
			printf("Phase %d - 8x8 block:\n",p);
			printBlock_uc(block);
		}
        // dct on the block
        dct(block, block_dct_out);
		if (dbg) {
			printf("Phase %d - DCT results of 8x8 block:\n", p);
			printBlock_f(block_dct_out);
		}
        // quant on the block
        quant(block_dct_out, quant_out);
		if (dbg) {
			printf("Quant results of 8x8 block:\n");
			printBlock_i(quant_out);
		}
        // zig zag on the block -> 64-d array
        zigzag(quant_out, zigzag_out);
        if (dbg) {
			printf("Print 64-d Zig Zag Scan Result:\n");
			///*
			for(int i=0;i<8*8;i++) {
				printf("%d ", zigzag_out[i]);
			}
			printf("\n\n");
		}
		//*/

        // RLE on 64-d array
        rle(zigzag_out, block_list); //, block_list
        if (dbg) {
            printf("Run Length Encoding Results of 64-d array:\n");
            printList_i(block_list);
        }
		
		// RLE -> zigzag out
        irle(block_list, zigzag_out);
        if (dbg) {
            printf("Inverse RLE Result:\n");
            for(int i=0;i<8*8;i++) {
                printf("%d ", zigzag_out[i]);
            }
            printf("\n\n");
        }


		// inverse zig zag -> 8x8 block
		izigzag(zigzag_out, izigzag_out);
		if (dbg) {
			printf("Output of Inverse ZigZag\n");
			printBlock_i(izigzag_out);
			printf("\n");
		}
		// inverse quantization on 8x8 block
		iquant(izigzag_out, iquant_out);
		if (dbg) {
			printf("Inverse Quant results of 8x8 block:\n");
			printBlock_i(iquant_out);
		}
		// inverse dct on 8x8 block
		idct(iquant_out, iblock_dct_out);
		if (dbg) {
			printf("Inverse DCT results of 8x8 block:\n");
			printBlock_uc(iblock_dct_out);
		}
		
		if (dbg)
			printf("Writing block back to memory\n");
			
		// write the decompressed block back to the img
		i_b = 0; j_b = 0;
		for (int i=(inc_row*7); i<(inc_row*7)+8; i++) {
            j_b=0;
            for (int j=(p*8); j<(p*8)+8; j++) {
				if (dbg)
					printf("i*m+j = %d | iblock_dct_out[i_b*8+j_b] = %u\n", i*m+j, iblock_dct_out[i_b*8+j_b]);
				
                imgDataOut[i*m+j] = iblock_dct_out[i_b*8+j_b]; //fill the img with the decompressed block
				
				if (dbg)
					printf("img[i*m+j] = %u\n\n", img[i*m+j]);
				
                j_b++;
            }
            i_b++; 
        }
		if (dbg)
			printf("--------------------------------------------------------------------\n");
		
		if (dbg) {
			if (p == 3)
				break;
		}
	}
}


/*
Timing class 'Elapsed' from https://www.codeproject.com/Tips/5254574/High-Resolution-Timing
Pasted source here to avoid claim that this is my code, it is not. This class is directly copy
	pasted from the above link.
*/
///*
class Elapsed
{
public :
    Elapsed()        // constructor
    {
        // get the frequency of the performance counter and determine its period

        LARGE_INTEGER li;
        if( QueryPerformanceFrequency( &li ) )
        {
            m_Available = true;
            m_Frequency = li.QuadPart;
            m_Period = 1.0 / (double)m_Frequency;    // period is in seconds
        }
    }

    // obtain elapsed time in seconds

    inline double Since( double begin=0 )
    {
        // get current performance counter value, convert to seconds,
        // and return the difference from the begin argument

        LARGE_INTEGER endtime;
        QueryPerformanceCounter( &endtime );
        return ( endtime.QuadPart * m_Period ) - begin;
    }

    // returns true if the counter is available

    bool    IsAvailable() const              { return m_Available; }

    // return the counter frequency

    INT64   GetFrequency() const             { return m_Frequency; }

    // return the period of the counter

    double  GetPeriod() const                { return m_Period; }

protected :
    bool    m_Available                      { false };
    double  m_Period                         { 0 };
    INT64   m_Frequency                      { 0 };
};
//*/

int main()
{
    // precompute the cosk and cosl for dct
    float* cosk = (float*)malloc(4096 * sizeof(float));
    float* cosl = (float*)malloc(4096 * sizeof(float));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < 8; k++)
                for (int l = 0; l < 8; l++) {
                    cosk[i * 512 + j * 64 + k * 8 + l] = cosf((2 * k + 1) * i * pi_math / (16));
                    cosl[i * 512 + j * 64 + k * 8 + l] = cosf((2 * l + 1) * j * pi_math / (16));
                }

    // determine run type for testing
    bool run_cpu_unit_test = 0;  // confirmed working for blocks 8x8, 16x16 and greater
    bool run_cpu_image_test = 1; // confirmed working for 240x240 and 512x512 Lena 

    bool run_gpu_unit_test = 0;
    bool run_gpu_image_test = 1;

    // Run the CPU Based Unit Test
    if (run_cpu_unit_test)
        cpu_unit_test();

    // Run the GPU Based Unit Test
    if (run_gpu_unit_test)
        gpu_unit_test(cosk, cosl, true);

    if (run_cpu_image_test || run_gpu_image_test) {
        // read input image from file
        const char filename[] = "C:/Users/blake/OneDrive/Desktop/GPU_Project/JPEG_Codec/dat/testimage2.png";
        int x_cols = 0; int y_rows = 0; int n_pixdepth = 0;
        unsigned char* imgData = stbi_load(filename, &x_cols, &y_rows, &n_pixdepth, 1);
        unsigned char* imgDataOut = (unsigned char*)malloc(y_rows * x_cols * sizeof(unsigned char)); // img dev from gpu to host
        unsigned char* imgDataOut_host = (unsigned char*)malloc(y_rows * x_cols * sizeof(unsigned char)); // img dev from gpu to host

        // Check that file exists
        FILE* infile = fopen(filename, "r");
        if (infile != NULL) printf("File found\n");
        else printf("File not found\n");
        fclose(infile);

        Elapsed timer;

        if (run_gpu_image_test) {
            // GPU Code
            printf("Running GPU JPEG Compression on sample %dx%d image:\n", y_rows, x_cols);
            std::vector< std::list<char>> encoded(y_rows*x_cols/64);

            // timer for gpu
            double start_gpu = timer.Since(0);
            gpu 
            // call the parallel codec
            //jpeg_codec_parallel(imgData, encoded, imgDataOut, y_rows, x_cols, false);
            jpeg_codec_parallel_rr(imgData, encoded, imgDataOut, cosk, cosl, y_rows, x_cols, false);

            // ping timer again for elapsed gpu time
            double elapsed_gpu = timer.Since(start_gpu); 

            // save the gpu parallel version of the reconstructed image
            const char imgFileOut1[] = "C:/Users/blake/OneDrive/Desktop/GPU_Project/JPEG_Codec/dat/outimage2_gpu.png";
            stbi_write_png(imgFileOut1, x_cols, y_rows, 1, imgDataOut, x_cols * n_pixdepth);

            printf("Parallel Code took: %lf seconds  |  %lf microseconds\n\n", (elapsed_gpu), (1.0E6 * elapsed_gpu));
        }

        if (run_cpu_image_test) {
            // host code
            const int m_img = y_rows;
            const int n_img = x_cols;
            printf("Running CPU JPEG Compression on sample %dx%d image:\n", m_img, n_img);

            // timer for cpu
            double start_cpu = timer.Since(0);

            // run the host sequential cpu code
            jpeg_codec_sequential(imgData, imgDataOut_host, m_img, n_img);

            // ping timer again for elapsed cpu time
            double elapsed_cpu = timer.Since(start_cpu);

            // save the cpu host version of the compressed image
            const char imgFileOut2[] = "C:/Users/blake/OneDrive/Desktop/GPU_Project/JPEG_Codec/dat/outimage2_cpu.png";
            stbi_write_png(imgFileOut2, x_cols, y_rows, 1, imgDataOut_host, x_cols * n_pixdepth);

            printf("Sequential Code took: %lf seconds  |  %lf microseconds\n\n", (elapsed_cpu), (1.0E6 * elapsed_cpu));
        }

        // if both ran, compare them
        if (run_cpu_image_test && run_gpu_image_test) {
            // compare outputs with simple MSE calculation
            int px_diff_mae = 0;
            for (int i = 0; i < x_cols * y_rows; i++)
                px_diff_mae += abs(imgDataOut[i] - imgDataOut_host[i]);
            printf("MAE between CPU and GPU versions of reconstructed images is %d\n", px_diff_mae/(x_cols*y_rows));
        }
        free(imgData);
        free(imgDataOut);
        free(imgDataOut_host);
    }
    free(cosk);
    free(cosl);
    return 0;
}
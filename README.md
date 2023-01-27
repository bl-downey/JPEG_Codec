# JPEG Codec

This project was the final project for The University of Washington's Professional Master's Program GPU Programming course.

## Introduction

GPU Parallel Processing allows for the computational speed up of many problems, one problem
of interest to me is parallel processing of image encoding and decoding. I think this is an
interesting application space as data transfer of images can take a lot of time, and if the images
need to be compressed on the way out and decompressed on the way in, having this process
be an efficient one is important.

Because of this need for fast data transfer of images, we need a way to parallelize the
compression. For this project, I chose JPEG compression and decompression (JPEG codec) for
this application. JPEG codec is well suited for GPU computing for a few reasons. In Figure 1
below we see the multiple compression stages of a JPEG Encoder, and in Figure 2 we can note
the steps of the image reconstruction via the JPEG Decoder. Because the mathematical
process of the DCT, quantization and zig zag algorithms are repeated for all 8x8 blocks in the
image, this brings about the opportunity for parallel processing

![image](https://user-images.githubusercontent.com/72525765/215004074-9ead30b7-ff14-4656-aecb-1e0a8550f069.png)

## Methods

The serial CPU JPEG codec was my first task and from there I utilized the mathematical and
algorithmic functions to convert the code to my parallel GPU version. Creating the GPU kernels
from the CPU functions was harder than expected for certain kernels as access patterns were
tricky. My parallel implementation has two versions (variations to dct and idct kernels), one
where I compute the cosine math in the kernel, and one where I precompute and then read from
global memory instead, an analysis of why and how it performed can be found in the report.

## Results

The dat/ folder contains the input images used, and the CPU/GPU output versions. As shown below, the CPU/GPU versions are identical. Check out page 10 of the report to see various timing results and the speed up of the DCT and iDCT kernels when using a precomputed method to lower the register count and increase thread count!

![outimage2_cpu](https://user-images.githubusercontent.com/72525765/215004552-114647c6-db5b-4762-8536-b5665cb12493.png)
Figure 3: CPU Version
![outimage2_gpu](https://user-images.githubusercontent.com/72525765/215004555-abb139b4-85d9-4c38-a22d-fe523908c05c.png)
Figure 4: GPU Version

## The code
The codec.cu file contains the functions to run the serial and parallel codecs, and different units tests that were used to ensure proper functionality at each step, as well as my main function. Inside utils/ you can find the supporting functions for the JPEG serial and parallel codecs.

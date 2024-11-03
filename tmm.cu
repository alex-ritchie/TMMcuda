#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

// Optimized CUDA Kernel for matrix multiplication with casting-based clamping
__global__ void tmm(int8_t* A, uint8_t* B_compressed, int8_t* C, int A_rows, int A_cols, int B_cols) {
    __shared__ int8_t As[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for tile of A
    __shared__ int8_t Bs_pos[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for positive values of B
    __shared__ int8_t Bs_neg[BLOCK_SIZE][BLOCK_SIZE];  // Shared memory for negative values of B

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int8_t sum = 0;

    // Loop over tiles of A and B
    for (int tileIdx = 0; tileIdx < (A_cols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tileIdx) {
        // Load A's tile into shared memory
        if (row < A_rows && (tileIdx * BLOCK_SIZE + threadIdx.x) < A_cols) {
            As[threadIdx.y][threadIdx.x] = A[row * A_cols + tileIdx * BLOCK_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load B's tile into shared memory and split into positive and negative parts
        if (col < B_cols && (tileIdx * BLOCK_SIZE + threadIdx.y) < A_cols) {
            uint8_t compressedB = B_compressed[(tileIdx * BLOCK_SIZE + threadIdx.y) * B_cols + col];
            if (compressedB == 1) {
                Bs_pos[threadIdx.y][threadIdx.x] = 1;
                Bs_neg[threadIdx.y][threadIdx.x] = 0;
            } else if (compressedB == 2) {  // -1 encoded as 2
                Bs_pos[threadIdx.y][threadIdx.x] = 0;
                Bs_neg[threadIdx.y][threadIdx.x] = 1;
            } else {  // compressedB == 0
                Bs_pos[threadIdx.y][threadIdx.x] = 0;
                Bs_neg[threadIdx.y][threadIdx.x] = 0;
            }
        } else {
            Bs_pos[threadIdx.y][threadIdx.x] = 0;
            Bs_neg[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Perform accumulation for positive and negative values of B with casting-based clamping
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            int16_t temp_add = (int16_t)As[threadIdx.y][k] * Bs_pos[k][threadIdx.x];
            int16_t temp_sub = (int16_t)As[threadIdx.y][k] * Bs_neg[k][threadIdx.x];

            // Accumulate into int8_t with implicit clamping via casting
            sum = static_cast<int8_t>(static_cast<int16_t>(sum) + temp_add);
            sum = static_cast<int8_t>(static_cast<int16_t>(sum) - temp_sub);
        }

        __syncthreads();
    }

    // Store the result
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = sum;
    }
}


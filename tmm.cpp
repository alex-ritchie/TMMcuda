#include <torch/extension.h>
#include <cuda_runtime.h>

// Declare the CUDA kernel function
__global__ void tmm(int8_t* A, uint8_t* B_compressed, int8_t* C, int A_rows, int A_cols, int B_cols);

// Wrapper function to call the CUDA kernel from PyTorch
torch::Tensor tmm(torch::Tensor A, torch::Tensor B_compressed, int A_rows, int A_cols, int B_cols) {
    // Check input types to ensure they are CUDA tensors
    TORCH_CHECK(A.is_cuda(), "Matrix A must be a CUDA tensor");
    TORCH_CHECK(B_compressed.is_cuda(), "Matrix B_compressed must be a CUDA tensor");

    // Allocate output tensor C with int8 dtype
    auto C = torch::zeros({A_rows, B_cols}, torch::kInt8).cuda();

    // Define the block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((B_cols + 15) / 16, (A_rows + 15) / 16);

    // Launch the CUDA kernel
    tmm<<<gridDim, blockDim>>>(
        A.data_ptr<int8_t>(),
        B_compressed.data_ptr<uint8_t>(),
        C.data_ptr<int8_t>(),
        A_rows,
        A_cols,
        B_cols
    );

    return C;
}

// Bind the function to Python using PyTorch's extension API
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_matrix_multiply", &tmm, "Matrix multiplication with int8 A and 2-bit B (CUDA)");
}


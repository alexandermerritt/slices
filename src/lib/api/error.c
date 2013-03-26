#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Error Management
//===----------------------------------------------------------------------===//

cudaError_t cuda_err = cudaSuccess;

cudaError_t assm_cudaGetLastError(void)
{
    return cuda_err;
}

const char* assm_cudaGetErrorString(cudaError_t error)
{
    return bypass.cudaGetErrorString(error);
}

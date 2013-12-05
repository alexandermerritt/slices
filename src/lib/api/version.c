#include "glob.h"
#include <precudart.h>

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Version Management
//===----------------------------------------------------------------------===//

cudaError_t
assm_cudaDriverGetVersion(int *v)
{
    FUNC_SETUP_CERR;
    if (!VGPU_IS_LOCAL(tinfo->vgpu))
        *((char*)buf) = 0; // shut up compiler
    cerr = bypass.cudaDriverGetVersion(v);
    return cerr;
}

cudaError_t
assm_cudaRuntimeGetVersion(int *v)
{
    FUNC_SETUP_CERR;
    if (!VGPU_IS_LOCAL(tinfo->vgpu))
        *((char*)buf) = 0; // shut up compiler
    cerr = bypass.cudaRuntimeGetVersion(v);
    return cerr;
}


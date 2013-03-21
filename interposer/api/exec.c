#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Execution Control
//===----------------------------------------------------------------------===//

cudaError_t
assm_cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaConfigureCall(buf, gridDim, blockDim, sharedMem, stream);
        rpc_ops.configureCall(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t
assm_cudaLaunch(const char* entry)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaLaunch(entry);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaLaunch(buf, entry);
        rpc_ops.launch(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t
assm_cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetupArgument(arg, size, offset);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetupArgument(buf, ((struct cuda_packet*)buf) + 1,
                arg, size, offset);
        rpc_ops.setupArgument(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


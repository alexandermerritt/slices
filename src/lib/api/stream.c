#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Stream Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaStreamCreate(cudaStream_t *pStream)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.cudaStreamCreate(pStream);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaStreamCreate(buf);
        rpc_ops.streamCreate(buf, NULL, rpc(tinfo));
        extract_cudaStreamCreate(buf, pStream); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaStreamSynchronize(cudaStream_t stream)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.cudaStreamSynchronize(stream);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaStreamSynchronize(buf, stream);
        rpc_ops.streamSynchronize(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


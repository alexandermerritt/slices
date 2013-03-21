#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Thread Management [DEPRECATED]
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaThreadExit(void)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaThreadExit();
    } else {
        init_buf(&buf, tinfo);
        pack_cudaThreadExit(buf);
        rpc_ops.threadExit(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaThreadSynchronize(void)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaThreadSynchronize();
    } else {
        init_buf(&buf, tinfo);
        pack_cudaThreadSynchronize(buf);
        rpc_ops.threadSynchronize(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


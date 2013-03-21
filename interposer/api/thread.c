#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Thread Management [DEPRECATED]
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaThreadExit(struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaThreadExit();
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaThreadExit(buf);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.threadExit(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaThreadSynchronize(struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaThreadSynchronize();
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaThreadSynchronize(buf);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.threadSynchronize(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}


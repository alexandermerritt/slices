#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Stream Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaStreamCreate(cudaStream_t *pStream,
        struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.cudaStreamCreate(pStream);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaStreamCreate(buf);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.streamCreate(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        extract_cudaStreamCreate(buf, pStream); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaStreamSynchronize(cudaStream_t stream,
        struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.cudaStreamSynchronize(stream);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaStreamSynchronize(buf, stream);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.streamSynchronize(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}


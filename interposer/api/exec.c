#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Execution Control
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaConfigureCall(buf, gridDim, blockDim, sharedMem, stream);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.configureCall(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaLaunch(const char* entry, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaLaunch(entry);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaLaunch(buf, entry);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.launch(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaSetupArgument(const void *arg, size_t size, size_t offset,
        struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetupArgument(arg, size, offset);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetupArgument(buf, ((struct cuda_packet*)buf) + 1,
                arg, size, offset);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.setupArgument(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}


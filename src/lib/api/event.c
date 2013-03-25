#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Event Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaEventCreate(cudaEvent_t * eventPtr)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventCreate(eventPtr);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventCreate(buf);
        rpc_ops.eventCreate(buf, NULL, rpc(tinfo));
        extract_cudaEventCreate(buf, eventPtr);
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventCreateWithFlags(cudaEvent_t * eventPtr,
        unsigned int flags)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventCreateWithFlags(eventPtr, flags);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventCreateWithFlags(buf, flags);
        rpc_ops.eventCreateWithFlags(buf, NULL, rpc(tinfo));
        extract_cudaEventCreateWithFlags(buf, eventPtr);
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventRecord(event, stream);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventRecord(buf, event, stream);
        rpc_ops.eventRecord(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventQuery(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventQuery(event);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventQuery(buf, event);
        rpc_ops.eventQuery(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventSynchronize(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventSynchronize(event);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventSynchronize(buf, event);
        rpc_ops.eventSynchronize(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventDestroy(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventDestroy(event);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventDestroy(buf, event);
        rpc_ops.eventDestroy(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventElapsedTime(float * ms,
        cudaEvent_t start, cudaEvent_t end)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventElapsedTime(ms, start, end);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaEventElapsedTime(buf, start, end);
        rpc_ops.eventElapsedTime(buf, NULL, rpc(tinfo));
        extract_cudaEventElapsedTime(buf, ms);
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


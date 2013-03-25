#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Event Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaEventCreate(cudaEvent_t * event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventCreate(event);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventCreateWithFlags(event, flags);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventRecord(event, stream);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventQuery(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventQuery(event);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventSynchronize(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventSynchronize(event);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventDestroy(cudaEvent_t event)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventDestroy(event);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaEventElapsedTime(ms, start, end);
    } else {
        BUG("not implemented");
        init_buf(&buf, tinfo);
        //pack_cudaMallocHost(buf, size);
        //rpc_ops.mallocHost(buf, NULL, rpc(tinfo));
        //extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        //cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


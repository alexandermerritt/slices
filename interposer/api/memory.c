#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Memory Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaFree(void * devPtr, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFree(devPtr);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaFree(buf, devPtr);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.free(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaFreeArray(struct cudaArray * array, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFreeArray(array);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaFreeArray(buf, array);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.freeArray(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaFreeHost(void *ptr, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFreeHost(ptr);
        TIMER_END(t, lat->lib.wait);
    } else {
        buf = NULL; /* hush compiler, hush */
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        if (ptr) free(ptr);
        TIMER_END(t, lat->lib.wait);
        cerr = cudaSuccess;
        /*LAT_UPDATE(lat, buf);*/
    }
    return cerr;
}

/* This function, according to the NVIDIA CUDA API specifications, seems to just
 * be a combined malloc+mlock that is additionally made visible to the CUDA
 * runtime and NVIDIA driver to optimize memory movements carried out in
 * subsequent calls to cudaMemcpy. There's no point in forwarding this function:
 * cudaHostAlloc returns an application virtual address, it is useless in the
 * application if RPC'd.
 *
 * We ignore flags for now, it only specifies performance not correctness.
 */
cudaError_t assm_cudaHostAlloc(void **pHost, size_t size, unsigned int flags,
        struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaHostAlloc(pHost, size, flags);
        TIMER_END(t, lat->lib.wait);
    } else {
        /*init_buf(&buf, tinfo);*/
        buf = NULL; /* hush compiler, don't cry */
        *pHost = malloc(size);
        if (!*pHost) {
            fprintf(stderr, "> Out of memory: %s\n", __func__);
            return cudaErrorMemoryAllocation;
        }
        TIMER_END(t, lat->lib.wait);
        cerr = cudaSuccess;
        /*LAT_UPDATE(lat, buf);*/
    }
    return cerr;
}

cudaError_t assm_cudaMalloc(void **devPtr, size_t size, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMalloc(devPtr, size);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMalloc(buf, size);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.malloc(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMallocArray(
        struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc,
        size_t width, size_t height, unsigned int flags, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMallocArray(array, desc, width, height, flags);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMallocArray(buf, desc, width, height, flags);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.mallocArray(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        extract_cudaMallocArray(buf, array); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMallocPitch(devPtr, pitch, width, height);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMallocPitch(buf, width, height);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.mallocPitch(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        extract_cudaMallocPitch(buf, devPtr, pitch); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpy(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpy(dst, src, count, kind);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpy(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyH2D(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToHost:
                rpc_ops.memcpyD2H(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToDevice:
                rpc_ops.memcpyD2D(buf, NULL, rpc(tinfo)); break;
			default:
				fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
				abort();
        }
        TIMER_END(t, lat->lib.wait);
        extract_cudaMemcpy(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyAsync(dst, src, count, kind, stream);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyAsync(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind, stream);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyAsyncH2D(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToHost:
                rpc_ops.memcpyAsyncD2H(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToDevice:
                rpc_ops.memcpyAsyncD2D(buf, NULL, rpc(tinfo)); break;
			default:
				fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
				abort();
        }
        TIMER_END(t, lat->lib.wait);
        extract_cudaMemcpyAsync(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind, stream); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count,
        size_t offset, enum cudaMemcpyKind kind, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyFromSymbol(buf, ((struct cuda_packet*)buf) + 1,
                dst, symbol, count, offset, kind);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyDeviceToHost:
                rpc_ops.memcpyFromSymbolD2H(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        TIMER_END(t, lat->lib.wait);
        extract_cudaMemcpyFromSymbol(buf, ((struct cuda_packet*)buf) + 1,
                dst, symbol, count, offset, kind); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToArray( struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToArray(buf, ((struct cuda_packet*)buf) + 1,
                dst, wOffset, hOffset, src, count, kind);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToArrayH2D(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToDevice:
                rpc_ops.memcpyToArrayD2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToSymbol(const char *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToSymbol(symbol, src, count, offset, kind);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToSymbol(buf, ((struct cuda_packet*)buf) + 1,
                symbol, src, count, offset, kind);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToSymbolH2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToSymbolAsync(const char *symbol, const void *src,
        size_t count, size_t offset,
        enum cudaMemcpyKind kind, cudaStream_t stream, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToSymbol(symbol, src, count, offset, kind);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToSymbolAsync(buf, ((struct cuda_packet*)buf) + 1,
                symbol, src, count, offset, kind, stream);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToSymbolAsyncH2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

/* XXX
 * If the assembly vgpu has a smaller amount of memory than the physical GPU, it
 * should return the difference in that share.
 */
cudaError_t assm_cudaMemGetInfo(size_t *free, size_t *total, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemGetInfo(free, total);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemGetInfo(buf);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.memGetInfo(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        extract_cudaMemGetInfo(buf, free, total); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemset(void *devPtr, int value, size_t count, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemset(devPtr, value, count);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemset(buf, devPtr, value, count);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.memset(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}


#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Memory Management
//===----------------------------------------------------------------------===//

cudaError_t assm_cudaFree(void * devPtr)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFree(devPtr);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaFree(buf, devPtr);
        rpc_ops.free(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaFreeArray(struct cudaArray * array)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFreeArray(array);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaFreeArray(buf, array);
        rpc_ops.freeArray(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaFreeHost(void *ptr)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaFreeHost(ptr);
    } else {
        buf = NULL; /* hush compiler, hush */
        if (ptr) free(ptr);
        cerr = cudaSuccess;
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
cudaError_t assm_cudaHostAlloc(void **hostPtr, size_t size, unsigned int flags)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaHostAlloc(hostPtr, size, flags);
    } else {
        buf = NULL; /* hush compiler, don't cry */
        cerr = cudaSuccess;
		if (!(*hostPtr = malloc(size)))
			cerr = cudaErrorMemoryAllocation;
    }
    return cerr;
}

// this is the 'old' version of the newer cudaHostAlloc
cudaError_t assm_cudaMallocHost(void **hostPtr, size_t size)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMallocHost(hostPtr, size);
    } else {
        buf = NULL; /* hush compiler, don't cry */
        cerr = cudaSuccess;
		if (!(*hostPtr = malloc(size)))
			cerr = cudaErrorMemoryAllocation;
    }
    return cerr;
}

cudaError_t assm_cudaMalloc(void **devPtr, size_t size)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMalloc(devPtr, size);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMalloc(buf, size);
        rpc_ops.malloc(buf, NULL, rpc(tinfo));
        extract_cudaMalloc(buf, devPtr); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMallocArray(
        struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc,
        size_t width, size_t height, unsigned int flags)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMallocArray(array, desc, width, height, flags);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMallocArray(buf, desc, width, height, flags);
        rpc_ops.mallocArray(buf, NULL, rpc(tinfo));
        extract_cudaMallocArray(buf, array); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMallocPitch(devPtr, pitch, width, height);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMallocPitch(buf, width, height);
        rpc_ops.mallocPitch(buf, NULL, rpc(tinfo));
        extract_cudaMallocPitch(buf, devPtr, pitch); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpy(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpy(dst, src, count, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpy(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind);
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
        extract_cudaMemcpy(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyAsync(dst, src, count, kind, stream);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyAsync(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind, stream);
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
        extract_cudaMemcpyAsync(buf, ((struct cuda_packet*)buf) + 1,
                dst, src, count, kind, stream); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count,
        size_t offset, enum cudaMemcpyKind kind)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyFromSymbol(buf, ((struct cuda_packet*)buf) + 1,
                dst, symbol, count, offset, kind);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyDeviceToHost:
                rpc_ops.memcpyFromSymbolD2H(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        extract_cudaMemcpyFromSymbol(buf, ((struct cuda_packet*)buf) + 1,
                dst, symbol, count, offset, kind); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToArray( struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToArray(buf, ((struct cuda_packet*)buf) + 1,
                dst, wOffset, hOffset, src, count, kind);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToArrayH2D(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToDevice:
                rpc_ops.memcpyToArrayD2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToSymbol(const char *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToSymbol(symbol, src, count, offset, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToSymbol(buf, ((struct cuda_packet*)buf) + 1,
                symbol, src, count, offset, kind);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToSymbolH2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpyToSymbolAsync(const char *symbol, const void *src,
        size_t count, size_t offset,
        enum cudaMemcpyKind kind, cudaStream_t stream)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpyToSymbol(symbol, src, count, offset, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpyToSymbolAsync(buf, ((struct cuda_packet*)buf) + 1,
                symbol, src, count, offset, kind, stream);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpyToSymbolAsyncH2D(buf, NULL, rpc(tinfo)); break;
            default: abort();
        }
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemcpy2D(void* dst, size_t dpitch, const void* src,
        size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemcpy2D(buf, ((struct cuda_packet*)buf) + 1,
                dst, dpitch, src, spitch, width, height, kind);
        switch (kind) {
            case cudaMemcpyHostToHost:
                return cudaSuccess; /* pack does a memcpy for us */
            case cudaMemcpyHostToDevice:
                rpc_ops.memcpy2DH2D(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToHost:
                rpc_ops.memcpy2DD2H(buf, NULL, rpc(tinfo)); break;
            case cudaMemcpyDeviceToDevice:
                rpc_ops.memcpy2DD2D(buf, NULL, rpc(tinfo)); break;
			default:
				fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
				abort();
        }
        /* XXX include in timing */
        extract_cudaMemcpy2D(buf, ((struct cuda_packet*)buf) + 1,
                dst, dpitch, src, spitch, width, height, kind);
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

/* XXX
 * If the assembly vgpu has a smaller amount of memory than the physical GPU, it
 * should return the difference in that share.
 */
cudaError_t assm_cudaMemGetInfo(size_t *free, size_t *total)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemGetInfo(free, total);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemGetInfo(buf);
        rpc_ops.memGetInfo(buf, NULL, rpc(tinfo));
        extract_cudaMemGetInfo(buf, free, total); /* XXX include in timing */
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t assm_cudaMemset(void *devPtr, int value, size_t count)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaMemset(devPtr, value, count);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaMemset(buf, devPtr, value, count);
        rpc_ops.memset(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


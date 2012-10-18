/**
 * @file cuda_interface.c
 * @date 2012-08-29
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief Interface used *only* by the interposer. remotesink uses cuda/rpc.c
 * and cuda/execute.c
 *
 * TODO It might be confusing to leave this file in backend/ (or confusing to
 * keep it regardless) since we don't really have a backend...
 */

#include <cuda/bypass.h>
#include <assembly.h>
#include "internals.h"

/* Things to know
 * - thread/vgpu association; if none exist, must put thread into some vgpu
 * - if vgpu is remote or local
 * - where the marshaling region is, for remote vgpus
 */

/*-------------------------------------- INTERNAL STATE ----------------------*/

/* We assume only one assembly/process */
static asmid_t assm_id;
static struct assembly *assm;

/* association between an application thread and vgpu in the assembly */
/* assume one assembly is used for now */
struct tinfo
{
    bool valid;
    /* application state */
    pthread_t tid;
    /* vgpu state */
    struct vgpu_mapping *vgpu;
    /* marshaling state (only if remote) */
    void *buffer;
};
static int num_tids;
/* XXX limited to 32 threads per application process */
static struct tinfo tinfos[32];
static pthread_mutex_t tinfo_lock = PTHREAD_MUTEX_INITIALIZER;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

/* TODO need some way of specifying vgpu mapping */
/* TODO return vgpu_id */
/* TODO lock only if adding/removing entry, not when looking up */
/* TODO add only at first !valid slot, not keep appending at end */
static struct tinfo *__lookup(pthread_t tid)
{
    int i;
    struct tinfo *ret = NULL;
    pthread_mutex_lock(&tinfo_lock);
    for (i = 0; i < num_tids; i++) {
        if (!tinfos[i].valid)
            continue;
        if (0 != pthread_equal(tinfos[i].tid, tid))
            break;
    }
    if (i < num_tids) { /* found tid state (more likely) */
        ret = &tinfos[i];
    } else { /* not found, make new entry */
        ret = &tinfos[num_tids++];
        ret->valid = true;
        ret->tid = tid;
        /* TODO only allocate if vgpu thread picks is remote */
        ret->buffer = malloc(768 << 20);
        ret->vgpu = &assm->mappings[0]; /* default 0 until setDevice called */
        if (!ret->buffer) {
            fprintf(stderr, "Out of memory\n");
            abort();
        }
    }
    pthread_mutex_unlock(&tinfo_lock);
    BUG(!ret);
    return ret;
}

#define rpc(tinfo)          ((tinfo)->vgpu->rpc)

#define VGPU_IS_LOCAL(vgpu) ((vgpu)->fixation == VGPU_LOCAL)

/*-------------------------------------- MISC API ----------------------------*/

extern struct assembly * assembly_find(asmid_t id);

/* XXX must be called before first cuda call is made
 * Interposer connects to runtime and queries for an assembly. runtime exports
 * it and provides it to the interposer. interposer then tells us what the
 * assembly ID is that we are supposed to use. */
int assm_cuda_init(asmid_t id)
{
    memset(tinfos, 0, sizeof(tinfos));
    assm_id = id;
    assm = assembly_find(assm_id);
    BUG(!assm);
    return 0;
}

/* should be called when no more cuda calls are made by application */
int assm_cuda_tini(void)
{
    int i;
    for (i = 0; i < num_tids; i++)
        if (tinfos[i].valid && tinfos[i].buffer)
            free(tinfos[i].buffer);
    memset(tinfos, 0, sizeof(tinfos));
    num_tids = 0;
    assm = NULL;
    return 0;
}

/*-------------------------------------- INTERPOSING API ---------------------*/

/* lots of boilerplate code below here, tried to keep it short */

/*
 * Preprocessor magic to reduce typing
 */

/*
 * Function setup code; lookup thread-specific state (include in marshaling
 * time).
 */
#define FUNC_SETUP \
    void *buf = NULL; \
    struct tinfo *tinfo; \
    TIMER_DECLARE1(t); \
    TIMER_START(t); \
    tinfo = __lookup(pthread_self())
#define FUNC_SETUP_CERR \
    FUNC_SETUP; \
    cudaError_t cerr = cudaSuccess

/* initialize the buf ptr once thread state has been looked up */
static inline void
init_buf(void **buf, struct tinfo *tinfo)
{
    *buf = tinfo->buffer;
    memset(*buf, 0, sizeof(struct cuda_packet));
}

//
// Thread Management API
//

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

//
// Device Managment API
//

cudaError_t assm_cudaGetDevice(int *device, struct rpc_latencies *lat)
{
    struct tinfo *tinfo = __lookup(pthread_self());
    *device = tinfo->vgpu->vgpu_id;
    return cudaSuccess;
}

cudaError_t assm_cudaGetDeviceCount(int *count, struct rpc_latencies *lat)
{
    *count = assm->num_gpus;
    return cudaSuccess;
}

cudaError_t assm_cudaGetDeviceProperties(struct cudaDeviceProp *prop,int device,
        struct rpc_latencies *lat)
{
    FUNC_SETUP;
    init_buf(&buf, tinfo);
    if (device < 0 || device >= assm->num_gpus)
        return cudaErrorInvalidDevice;
    /* don't need to translate vgpu to pgpu ID since the index into mappings is
     * virtual IDs already */
    memcpy(prop, &assm->mappings[device].cudaDevProp, sizeof(*prop));
    TIMER_END(t, lat->lib.wait); /* XXX ?? */
    printd(DBG_DEBUG, "name=%s\n", assm->mappings[device].cudaDevProp.name);
    return cudaSuccess;
}

cudaError_t assm_cudaSetDevice(int device, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (device >= assm->num_gpus)
        return cudaErrorInvalidDevice;
    tinfo->vgpu = &assm->mappings[device];
    /* translate vgpu device ID to physical ID */
    device = tinfo->vgpu->pgpu_id;
    /* let it pass through so the driver makes the association */
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetDevice(device);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        /* XXX should we send a flushing call to clear the batch queue here? */
        pack_cudaSetDevice(buf, device);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.setDevice(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaSetDeviceFlags(unsigned int flags, struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetDeviceFlags(flags);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetDeviceFlags(buf, flags);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.setDeviceFlags(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

cudaError_t assm_cudaSetValidDevices(int *device_arr, int len,
        struct rpc_latencies *lat)
{
    FUNC_SETUP_CERR;

    /* XXX This function is ignored from within cuda_runtime.c */

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetValidDevices(device_arr, len);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetValidDevices(buf, ((struct cuda_packet*)buf) + 1,
                device_arr, len);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.setValidDevices(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        cerr = cpkt_ret_err(buf);
        LAT_UPDATE(lat, buf);
    }
    return cerr;
}

//
// Stream Management API
//

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

//
// Execution Control API
//

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

//
// Memory Management API
//

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

//
// Undocumented API
//

void** assm__cudaRegisterFatBinary(void *cubin, struct rpc_latencies *lat)
{
    /* TODO duplicate call */
    FUNC_SETUP;
    void** ret;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        ret = bypass.__cudaRegisterFatBinary(cubin);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFatBinary(buf, (buf + sizeof(struct cuda_packet)), cubin);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.registerFatBinary(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        ret = cpkt_ret_hdl(buf);
        LAT_UPDATE(lat, buf);
    }
    return ret;
}

void assm__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFunction(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
                tid, bid, bDim, gDim, wSize);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.registerFunction(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

void assm__cudaUnregisterFatBinary(void** fatCubinHandle, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaUnregisterFatBinary(fatCubinHandle);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaUnregisterFatBinary(buf, fatCubinHandle);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.unregisterFatBinary(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

void assm__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char
        *deviceAddress, const char *deviceName, int ext, int vsize,
        int constant, int global, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
                deviceName, ext, vsize, constant, global);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterVar(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                vsize, constant, global);
        TIMER_END(t, lat->lib.setup);
        rpc_ops.registerVar(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

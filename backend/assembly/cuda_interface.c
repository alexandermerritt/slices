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
struct tid_vgpu
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
static struct tid_vgpu tid_vgpus[32];
static pthread_mutex_t tid_vgpu_lock = PTHREAD_MUTEX_INITIALIZER;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

/* TODO need some way of specifying vgpu mapping */
/* TODO return vgpu_id */
/* TODO lock only if adding/removing entry, not when looking up */
/* TODO add only at first !valid slot, not keep appending at end */
static struct tid_vgpu *__lookup(pthread_t tid)
{
    int i;
    struct tid_vgpu *ret = NULL;
    pthread_mutex_lock(&tid_vgpu_lock);
    for (i = 0; i < num_tids; i++) {
        if (!tid_vgpus[i].valid)
            continue;
        if (0 != pthread_equal(tid_vgpus[i].tid, tid))
            break;
    }
    if (i < num_tids) { /* found tid state (more likely) */
        ret = &tid_vgpus[i];
    } else { /* not found, make new entry */
        ret = &tid_vgpus[num_tids++];
        ret->valid = true;
        ret->tid = tid;
        /* TODO only allocate if vgpu thread picks is remote */
        ret->buffer = malloc(128UL << 20);
        ret->vgpu = &assm->mappings[0]; /* default 0 until setDevice called */
        if (!ret->buffer) {
            fprintf(stderr, "Out of memory\n");
            abort();
        }
    }
    pthread_mutex_unlock(&tid_vgpu_lock);
    BUG(!ret);
    return ret;
}

#define VGPU_IS_LOCAL(vgpu) ((vgpu)->fixation == VGPU_LOCAL)

/*-------------------------------------- MISC API ----------------------------*/

extern struct assembly * assembly_find(asmid_t id);

/* XXX must be called before first cuda call is made
 * Interposer connects to runtime and queries for an assembly. runtime exports
 * it and provides it to the interposer. interposer then tells us what the
 * assembly ID is that we are supposed to use. */
int assm_cuda_init(asmid_t id)
{
    memset(tid_vgpus, 0, sizeof(tid_vgpus));
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
        if (tid_vgpus[i].valid && tid_vgpus[i].buffer)
            free(tid_vgpus[i].buffer);
    memset(tid_vgpus, 0, sizeof(tid_vgpus));
    num_tids = 0;
    assm = NULL;
    return 0;
}

/*-------------------------------------- INTERPOSING API ---------------------*/

/* TODO cudaSetDevice must update is_local on tid_vgpu */

    /* Is vgpu local?
     *  yes - bypass
     *  no - marshal then call remote_ops
     */

cudaError_t assm_cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream)
{
    void *buf = NULL;
    struct tid_vgpu *tid_vgpu = __lookup(pthread_self());
    cudaError_t cerr;

    if (VGPU_IS_LOCAL(tid_vgpu->vgpu))
        cerr = bypass.cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
    else {
        buf = tid_vgpu->buffer;
        memset(buf, 0, sizeof(struct cuda_packet));
        pack_cudaConfigureCall(buf, gridDim, blockDim, sharedMem, stream);
        rpc_ops.configureCall(buf, NULL, tid_vgpu->vgpu->rpc);
        cerr = ((struct cuda_packet*)buf)->ret_ex_val.err;
    }
    return cerr;
}

cudaError_t assm_cudaLaunch(const char* entry)
{
    return bypass.cudaLaunch(entry);
}

void** assm__cudaRegisterFatBinary(void *cubin)
{
    /* TODO duplicate call */
    return bypass.__cudaRegisterFatBinary(cubin);
}

void assm__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
    bypass.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
            deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

void assm__cudaUnregisterFatBinary(void** fatCubinHandle)
{
    bypass.__cudaUnregisterFatBinary(fatCubinHandle);
}


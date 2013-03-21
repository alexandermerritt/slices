/**
 * @file glob.h
 * @author Alexander Merritt, merritt.alex@gatech.edu
 * @desc Assembly interposer globals
 */

#ifndef INTERPOSER_ASSM_GLOB_INCLUDED
#define INTERPOSER_ASSM_GLOB_INCLUDED

#include <cuda/bypass.h>
#include <assembly.h>
#include "internals.h"

/* Things to know
 * - thread/vgpu association; if none exist, must put thread into some vgpu
 * - if vgpu is remote or local
 * - where the marshaling region is, for remote vgpus
 */

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

struct tinfo *__lookup(pthread_t tid);

#define rpc(tinfo)          ((tinfo)->vgpu->rpc)
#define VGPU_IS_LOCAL(vgpu) ((vgpu)->fixation == VGPU_LOCAL)

extern struct assembly * assembly_find(asmid_t id);

/*
 * Preprocessor magic to reduce typing
 */

#define FUNC_SETUP \
    void *buf = NULL; \
    struct tinfo *tinfo; \
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


// Device API
cudaError_t assm_cudaGetDevice(int*);
cudaError_t assm_cudaGetDeviceCount(int*);
cudaError_t assm_cudaGetDeviceProperties(struct cudaDeviceProp*, int);
cudaError_t assm_cudaSetDevice(int);
cudaError_t assm_cudaSetDeviceFlags(unsigned int);
cudaError_t assm_cudaSetValidDevices(int*, int);

// Execution API
cudaError_t assm_cudaConfigureCall(dim3, dim3, size_t, cudaStream_t);
cudaError_t assm_cudaLaunch(const char*);
cudaError_t assm_cudaSetupArgument(const void*, size_t, size_t);

// Hidden API
void ** assm__cudaRegisterFatBinary(void *cubin);
void assm__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
void assm__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char
        *deviceAddress, const char *deviceName, int ext, int vsize,
        int constant, int global);
void assm__cudaUnregisterFatBinary(void** fatCubinHandle);

// Memory API
cudaError_t assm_cudaFree(void * devPtr);
cudaError_t assm_cudaFreeArray(struct cudaArray * array);
cudaError_t assm_cudaFreeHost(void *ptr);
cudaError_t assm_cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
cudaError_t assm_cudaMalloc(void **devPtr, size_t size);
cudaError_t assm_cudaMallocArray(struct cudaArray **array,
        const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags);
cudaError_t assm_cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t assm_cudaMemcpy(void *dst, const void *src,
        size_t count, enum cudaMemcpyKind kind);
cudaError_t assm_cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t assm_cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count,
        size_t offset, enum cudaMemcpyKind kind);
cudaError_t assm_cudaMemcpyToArray( struct cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t assm_cudaMemcpyToSymbol(const char *symbol, const void *src,
        size_t count, size_t offset, enum cudaMemcpyKind kind);

// Stream API
cudaError_t assm_cudaStreamCreate(cudaStream_t *pStream);
cudaError_t assm_cudaStreamSynchronize(cudaStream_t stream);

// Thread API
cudaError_t assm_cudaThreadExit(void);
cudaError_t assm_cudaThreadSynchronize(void);

#endif

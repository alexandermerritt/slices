#include "glob.h"
#include <precudart.h>

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

void cudaJmpTblConstructor(void)
{
    preCudaJmp.cudaGetDevice                = assm_cudaGetDevice;
    preCudaJmp.cudaGetDeviceCount           = assm_cudaGetDeviceCount;
    preCudaJmp.cudaGetDeviceProperties      = assm_cudaGetDeviceProperties;
    preCudaJmp.cudaSetDevice                = assm_cudaSetDevice;
    preCudaJmp.cudaSetDeviceFlags           = assm_cudaSetDeviceFlags;
    preCudaJmp.cudaSetValidDevices          = assm_cudaSetValidDevices;

    preCudaJmp.cudaConfigureCall            = assm_cudaConfigureCall;
    preCudaJmp.cudaLaunch                   = assm_cudaLaunch;
    preCudaJmp.cudaSetupArgument            = assm_cudaSetupArgument;

    preCudaJmp.cudaEventCreate              = assm_cudaEventCreate;
    preCudaJmp.cudaEventCreateWithFlags     = assm_cudaEventCreateWithFlags;
    preCudaJmp.cudaEventDestroy             = assm_cudaEventDestroy;
    preCudaJmp.cudaEventElapsedTime         = assm_cudaEventElapsedTime;
    preCudaJmp.cudaEventRecord              = assm_cudaEventRecord;
    preCudaJmp.cudaEventSynchronize         = assm_cudaEventSynchronize;

    preCudaJmp.__cudaRegisterFatBinary      = assm__cudaRegisterFatBinary;
    preCudaJmp.__cudaRegisterFunction       = assm__cudaRegisterFunction;
    preCudaJmp.__cudaRegisterVar            = assm__cudaRegisterVar;
    preCudaJmp.__cudaUnregisterFatBinary    = assm__cudaUnregisterFatBinary;

    preCudaJmp.cudaFree                     = assm_cudaFree;
    preCudaJmp.cudaFreeArray                = assm_cudaFreeArray;
    preCudaJmp.cudaFreeHost                 = assm_cudaFreeHost;
    preCudaJmp.cudaHostAlloc                = assm_cudaHostAlloc;
    preCudaJmp.cudaMalloc                   = assm_cudaMalloc;
    preCudaJmp.cudaMallocHost               = assm_cudaMallocHost;
    preCudaJmp.cudaMallocArray              = assm_cudaMallocArray;
    preCudaJmp.cudaMallocPitch              = assm_cudaMallocPitch;
    preCudaJmp.cudaMemcpy                   = assm_cudaMemcpy;
    preCudaJmp.cudaMemcpyAsync              = assm_cudaMemcpyAsync;
    preCudaJmp.cudaMemcpyFromSymbol         = assm_cudaMemcpyFromSymbol;
    preCudaJmp.cudaMemcpyToArray            = assm_cudaMemcpyToArray;
    preCudaJmp.cudaMemcpyToSymbol           = assm_cudaMemcpyToSymbol;
    preCudaJmp.cudaMemcpyToSymbolAsync      = assm_cudaMemcpyToSymbolAsync;
    preCudaJmp.cudaMemcpy2D                 = assm_cudaMemcpy2D;
    preCudaJmp.cudaMemset                   = assm_cudaMemset;
    preCudaJmp.cudaMemGetInfo               = assm_cudaMemGetInfo;

    preCudaJmp.cudaStreamCreate             = assm_cudaStreamCreate;
    preCudaJmp.cudaStreamDestroy            = assm_cudaStreamDestroy;
    preCudaJmp.cudaStreamSynchronize        = assm_cudaStreamSynchronize;

    preCudaJmp.cudaThreadExit               = assm_cudaThreadExit;
    preCudaJmp.cudaThreadSynchronize        = assm_cudaThreadSynchronize;

    preCudaJmp.cudaGetErrorString           = assm_cudaGetErrorString;
    preCudaJmp.cudaGetLastError             = assm_cudaGetLastError;
}

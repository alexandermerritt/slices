/**
 * @file bypass.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 10, 2012
 * @brief Allow interposer to fall through to true CUDA implementation.
 */

#include <cuda/bypass.h> // include first, as _GNU_SOURCE must be defined for dlsym

#include <stdio.h>
#include <stdlib.h>
#include <debug.h>

// Globals

struct bypass bypass;

// Functions

void fill_bypass(struct bypass *bypass)
{
    char *dlerr;
    void *cuda_handle;

    dlerror(); // clear errors
    cuda_handle = dlopen("libcudart.so", RTLD_NOW);
    if ((dlerr = dlerror())) {
        fprintf(stderr, ">> DL ERROR: '%s'\n", dlerr);
        exit(1);
    }

	bypass->cudaBindTexture            =  dlsym(cuda_handle,  "cudaBindTexture");
	bypass->cudaBindTextureToArray     =  dlsym(cuda_handle,  "cudaBindTextureToArray");
	bypass->cudaConfigureCall          =  dlsym(cuda_handle,  "cudaConfigureCall");
	bypass->cudaCreateChannelDesc      =  dlsym(cuda_handle,  "cudaCreateChannelDesc");
	bypass->cudaDriverGetVersion       =  dlsym(cuda_handle,  "cudaDriverGetVersion");
	bypass->cudaFreeArray              =  dlsym(cuda_handle,  "cudaFreeArray");
	bypass->cudaFree                   =  dlsym(cuda_handle,  "cudaFree");
	bypass->cudaFreeHost               =  dlsym(cuda_handle,  "cudaFreeHost");
	bypass->cudaFuncGetAttributes      =  dlsym(cuda_handle,  "cudaFuncGetAttributes");
	bypass->cudaGetDeviceCount         =  dlsym(cuda_handle,  "cudaGetDeviceCount");
	bypass->cudaGetDevice              =  dlsym(cuda_handle,  "cudaGetDevice");
	bypass->cudaGetDeviceProperties    =  dlsym(cuda_handle,  "cudaGetDeviceProperties");
	bypass->cudaGetErrorString         =  dlsym(cuda_handle,  "cudaGetErrorString");
	bypass->cudaGetLastError           =  dlsym(cuda_handle,  "cudaGetLastError");
	bypass->cudaGetTextureReference    =  dlsym(cuda_handle,  "cudaGetTextureReference");
	bypass->cudaHostAlloc              =  dlsym(cuda_handle,  "cudaHostAlloc");
	bypass->cudaLaunch                 =  dlsym(cuda_handle,  "cudaLaunch");
	bypass->cudaMallocArray            =  dlsym(cuda_handle,  "cudaMallocArray");
	bypass->cudaMalloc                 =  dlsym(cuda_handle,  "cudaMalloc");
	bypass->cudaMallocPitch            =  dlsym(cuda_handle,  "cudaMallocPitch");
	bypass->cudaMemcpyAsync            =  dlsym(cuda_handle,  "cudaMemcpyAsync");
	bypass->cudaMemcpy                 =  dlsym(cuda_handle,  "cudaMemcpy");
	bypass->cudaMemcpyFromSymbol       =  dlsym(cuda_handle,  "cudaMemcpyFromSymbol");
	bypass->cudaMemcpyToArray          =  dlsym(cuda_handle,  "cudaMemcpyToArray");
	bypass->cudaMemcpyToSymbolAsync    =  dlsym(cuda_handle,  "cudaMemcpyToSymbolAsync");
	bypass->cudaMemcpyToSymbol         =  dlsym(cuda_handle,  "cudaMemcpyToSymbol");
	bypass->cudaMemGetInfo             =  dlsym(cuda_handle,  "cudaMemGetInfo");
	bypass->cudaMemset                 =  dlsym(cuda_handle,  "cudaMemset");
	bypass->__cudaRegisterFatBinary    =  dlsym(cuda_handle,  "__cudaRegisterFatBinary");
	bypass->__cudaRegisterFunction     =  dlsym(cuda_handle,  "__cudaRegisterFunction");
	bypass->__cudaRegisterTexture      =  dlsym(cuda_handle,  "__cudaRegisterTexture");
	bypass->__cudaRegisterVar          =  dlsym(cuda_handle,  "__cudaRegisterVar");
	bypass->cudaRuntimeGetVersion      =  dlsym(cuda_handle,  "cudaRuntimeGetVersion");
	bypass->cudaSetDevice              =  dlsym(cuda_handle,  "cudaSetDevice");
	bypass->cudaSetDeviceFlags         =  dlsym(cuda_handle,  "cudaSetDeviceFlags");
	bypass->cudaSetupArgument          =  dlsym(cuda_handle,  "cudaSetupArgument");
	bypass->cudaSetValidDevices        =  dlsym(cuda_handle,  "cudaSetValidDevices");
	bypass->cudaStreamCreate           =  dlsym(cuda_handle,  "cudaStreamCreate");
	bypass->cudaStreamDestroy          =  dlsym(cuda_handle,  "cudaStreamDestroy");
	bypass->cudaStreamQuery            =  dlsym(cuda_handle,  "cudaStreamQuery");
	bypass->cudaStreamSynchronize      =  dlsym(cuda_handle,  "cudaStreamSynchronize");
	bypass->cudaThreadExit             =  dlsym(cuda_handle,  "cudaThreadExit");
	bypass->cudaThreadSynchronize      =  dlsym(cuda_handle,  "cudaThreadSynchronize");
	bypass->__cudaUnregisterFatBinary  =  dlsym(cuda_handle,  "__cudaUnregisterFatBinary");

    BUG(!bypass->__cudaRegisterFatBinary);
}


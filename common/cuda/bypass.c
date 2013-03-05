/**
 * @file bypass.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 10, 2012
 * @brief Allow interposer to fall through to true CUDA implementation.
 */

#include <cuda/bypass.h> // include first, as _GNU_SOURCE must be defined for dlsym

#include <stdio.h>
#include <stdlib.h>

// Globals

struct bypass bypass;

// Functions

void fill_bypass(struct bypass *bypass)
{
	// damn compiler complains about void* casting (icc, not gcc)
	bypass->cudaBindTexture            =  (fn_cudaBindTexture)         	dlsym(RTLD_NEXT,  "cudaBindTexture");
	bypass->cudaBindTextureToArray     =  (fn_cudaBindTextureToArray)  	dlsym(RTLD_NEXT,  "cudaBindTextureToArray");
	bypass->cudaConfigureCall          =  (fn_cudaConfigureCall)       	dlsym(RTLD_NEXT,  "cudaConfigureCall");
	bypass->cudaCreateChannelDesc      =  (fn_cudaCreateChannelDesc)   	dlsym(RTLD_NEXT,  "cudaCreateChannelDesc");
	bypass->cudaDriverGetVersion       =  (fn_cudaDriverGetVersion)    	dlsym(RTLD_NEXT,  "cudaDriverGetVersion");
	bypass->cudaFreeArray              =  (fn_cudaFreeArray)           	dlsym(RTLD_NEXT,  "cudaFreeArray");
	bypass->cudaFree                   =  (fn_cudaFree)                	dlsym(RTLD_NEXT,  "cudaFree");
	bypass->cudaFreeHost               =  (fn_cudaFreeHost)            	dlsym(RTLD_NEXT,  "cudaFreeHost");
	bypass->cudaFuncGetAttributes      =  (fn_cudaFuncGetAttributes)   	dlsym(RTLD_NEXT,  "cudaFuncGetAttributes");
	bypass->cudaGetDeviceCount         =  (fn_cudaGetDeviceCount)      	dlsym(RTLD_NEXT,  "cudaGetDeviceCount");
	bypass->cudaGetDevice              =  (fn_cudaGetDevice)           	dlsym(RTLD_NEXT,  "cudaGetDevice");
	bypass->cudaGetDeviceProperties    =  (fn_cudaGetDeviceProperties) 	dlsym(RTLD_NEXT,  "cudaGetDeviceProperties");
	bypass->cudaGetErrorString         =  (fn_cudaGetErrorString)      	dlsym(RTLD_NEXT,  "cudaGetErrorString");
	bypass->cudaGetLastError           =  (fn_cudaGetLastError)        	dlsym(RTLD_NEXT,  "cudaGetLastError");
	bypass->cudaGetTextureReference    =  (fn_cudaGetTextureReference) 	dlsym(RTLD_NEXT,  "cudaGetTextureReference");
	bypass->cudaHostAlloc              =  (fn_cudaHostAlloc)           	dlsym(RTLD_NEXT,  "cudaHostAlloc");
	bypass->cudaLaunch                 =  (fn_cudaLaunch)              	dlsym(RTLD_NEXT,  "cudaLaunch");
	bypass->cudaMallocArray            =  (fn_cudaMallocArray)         	dlsym(RTLD_NEXT,  "cudaMallocArray");
	bypass->cudaMalloc                 =  (fn_cudaMalloc)              	dlsym(RTLD_NEXT,  "cudaMalloc");
	bypass->cudaMallocPitch            =  (fn_cudaMallocPitch)         	dlsym(RTLD_NEXT,  "cudaMallocPitch");
	bypass->cudaMemcpyAsync            =  (fn_cudaMemcpyAsync)         	dlsym(RTLD_NEXT,  "cudaMemcpyAsync");
	bypass->cudaMemcpy                 =  (fn_cudaMemcpy)              	dlsym(RTLD_NEXT,  "cudaMemcpy");
	bypass->cudaMemcpyFromSymbol       =  (fn_cudaMemcpyFromSymbol)    	dlsym(RTLD_NEXT,  "cudaMemcpyFromSymbol");
	bypass->cudaMemcpyToArray          =  (fn_cudaMemcpyToArray)       	dlsym(RTLD_NEXT,  "cudaMemcpyToArray");
	bypass->cudaMemcpyToSymbolAsync    =  (fn_cudaMemcpyToSymbolAsync) 	dlsym(RTLD_NEXT,  "cudaMemcpyToSymbolAsync");
	bypass->cudaMemcpyToSymbol         =  (fn_cudaMemcpyToSymbol)      	dlsym(RTLD_NEXT,  "cudaMemcpyToSymbol");
	bypass->cudaMemGetInfo             =  (fn_cudaMemGetInfo)          	dlsym(RTLD_NEXT,  "cudaMemGetInfo");
	bypass->cudaMemset                 =  (fn_cudaMemset)              	dlsym(RTLD_NEXT,  "cudaMemset");
	bypass->__cudaRegisterFatBinary    =  (fn__cudaRegisterFatBinary)  	dlsym(RTLD_NEXT,  "__cudaRegisterFatBinary");
	bypass->__cudaRegisterFunction     =  (fn__cudaRegisterFunction)   	dlsym(RTLD_NEXT,  "__cudaRegisterFunction");
	bypass->__cudaRegisterTexture      =  (fn__cudaRegisterTexture)    	dlsym(RTLD_NEXT,  "__cudaRegisterTexture");
	bypass->__cudaRegisterVar          =  (fn__cudaRegisterVar)        	dlsym(RTLD_NEXT,  "__cudaRegisterVar");
	bypass->cudaRuntimeGetVersion      =  (fn_cudaRuntimeGetVersion)   	dlsym(RTLD_NEXT,  "cudaRuntimeGetVersion");
	bypass->cudaSetDevice              =  (fn_cudaSetDevice)           	dlsym(RTLD_NEXT,  "cudaSetDevice");
	bypass->cudaSetDeviceFlags         =  (fn_cudaSetDeviceFlags)      	dlsym(RTLD_NEXT,  "cudaSetDeviceFlags");
	bypass->cudaSetupArgument          =  (fn_cudaSetupArgument)       	dlsym(RTLD_NEXT,  "cudaSetupArgument");
	bypass->cudaSetValidDevices        =  (fn_cudaSetValidDevices)     	dlsym(RTLD_NEXT,  "cudaSetValidDevices");
	bypass->cudaStreamCreate           =  (fn_cudaStreamCreate)        	dlsym(RTLD_NEXT,  "cudaStreamCreate");
	bypass->cudaStreamDestroy          =  (fn_cudaStreamDestroy)       	dlsym(RTLD_NEXT,  "cudaStreamDestroy");
	bypass->cudaStreamQuery            =  (fn_cudaStreamQuery)         	dlsym(RTLD_NEXT,  "cudaStreamQuery");
	bypass->cudaStreamSynchronize      =  (fn_cudaStreamSynchronize)   	dlsym(RTLD_NEXT,  "cudaStreamSynchronize");
	bypass->cudaThreadExit             =  (fn_cudaThreadExit)          	dlsym(RTLD_NEXT,  "cudaThreadExit");
	bypass->cudaThreadSynchronize      =  (fn_cudaThreadSynchronize)   	dlsym(RTLD_NEXT,  "cudaThreadSynchronize");
	bypass->__cudaUnregisterFatBinary  =  (fn__cudaUnregisterFatBinary)	dlsym(RTLD_NEXT,  "__cudaUnregisterFatBinary");

	if (!bypass->__cudaRegisterFatBinary) {
		fprintf(stderr, "Error: binary was not linked with libcudart. Relink code.\n");
		exit(1);
	}
}


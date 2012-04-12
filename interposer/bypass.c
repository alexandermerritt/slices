/**
 * @file bypass.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 10, 2012
 * @brief Allow interposer to fall through to true CUDA implementation.
 */

#include "bypass.h" // include first, as _GNU_SOURCE must be defined for dlsym

#include <stdio.h>
#include <stdlib.h>

// Globals

struct bypass bypass;

// Functions

void fill_bypass(struct bypass *bypass)
{
	bypass->cudaBindTexture = dlsym(RTLD_NEXT, "cudaBindTexture");
	bypass->cudaBindTextureToArray = dlsym(RTLD_NEXT, "cudaBindTextureToArray");
	bypass->cudaConfigureCall = dlsym(RTLD_NEXT, "cudaConfigureCall");
	bypass->cudaCreateChannelDesc = dlsym(RTLD_NEXT, "cudaCreateChannelDesc");
	bypass->cudaDriverGetVersion = dlsym(RTLD_NEXT, "cudaDriverGetVersion");
	bypass->cudaFreeArray = dlsym(RTLD_NEXT, "cudaFreeArray");
	bypass->cudaFree = dlsym(RTLD_NEXT, "cudaFree");
	bypass->cudaFreeHost = dlsym(RTLD_NEXT, "cudaFreeHost");
	bypass->cudaFuncGetAttributes = dlsym(RTLD_NEXT, "cudaFuncGetAttributes");
	bypass->cudaGetDeviceCount = dlsym(RTLD_NEXT, "cudaGetDeviceCount");
	bypass->cudaGetDevice = dlsym(RTLD_NEXT, "cudaGetDevice");
	bypass->cudaGetDeviceProperties = dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
	bypass->cudaGetErrorString = dlsym(RTLD_NEXT, "cudaGetErrorString");
	bypass->cudaGetLastError = dlsym(RTLD_NEXT, "cudaGetLastError");
	bypass->cudaGetTextureReference = dlsym(RTLD_NEXT, "cudaGetTextureReference");
	bypass->cudaHostAlloc = dlsym(RTLD_NEXT, "cudaHostAlloc");
	bypass->cudaLaunch = dlsym(RTLD_NEXT, "cudaLaunch");
	bypass->cudaMallocArray = dlsym(RTLD_NEXT, "cudaMallocArray");
	bypass->cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
	bypass->cudaMallocPitch = dlsym(RTLD_NEXT, "cudaMallocPitch");
	bypass->cudaMemcpyAsync = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
	bypass->cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
	bypass->cudaMemcpyFromSymbol = dlsym(RTLD_NEXT, "cudaMemcpyFromSymbol");
	bypass->cudaMemcpyToArray = dlsym(RTLD_NEXT, "cudaMemcpyToArray");
	bypass->cudaMemcpyToSymbolAsync = dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");
	bypass->cudaMemcpyToSymbol = dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
	bypass->cudaMemGetInfo = dlsym(RTLD_NEXT, "cudaMemGetInfo");
	bypass->cudaMemset = dlsym(RTLD_NEXT, "cudaMemset");
	bypass->cudaRuntimeGetVersion = dlsym(RTLD_NEXT, "cudaRuntimeGetVersion");
	bypass->cudaSetDevice = dlsym(RTLD_NEXT, "cudaSetDevice");
	bypass->cudaSetDeviceFlags = dlsym(RTLD_NEXT, "cudaSetDeviceFlags");
	bypass->cudaSetupArgument = dlsym(RTLD_NEXT, "cudaSetupArgument");
	bypass->cudaSetValidDevices = dlsym(RTLD_NEXT, "cudaSetValidDevices");
	bypass->cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
	bypass->cudaStreamDestroy = dlsym(RTLD_NEXT, "cudaStreamDestroy");
	bypass->cudaStreamQuery = dlsym(RTLD_NEXT, "cudaStreamQuery");
	bypass->cudaStreamSynchronize = dlsym(RTLD_NEXT, "cudaStreamSynchronize");
	bypass->cudaThreadExit = dlsym(RTLD_NEXT, "cudaThreadExit");
	bypass->cudaThreadSynchronize = dlsym(RTLD_NEXT, "cudaThreadSynchronize");

	bypass->__cudaRegisterFatBinary = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
	bypass->__cudaRegisterFunction = dlsym(RTLD_NEXT, "__cudaRegisterFunction");
	bypass->__cudaRegisterTexture = dlsym(RTLD_NEXT, "__cudaRegisterTexture");
	bypass->__cudaRegisterVar = dlsym(RTLD_NEXT, "__cudaRegisterVar");
	bypass->__cudaUnregisterFatBinary = dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");

	if (!bypass->__cudaRegisterFatBinary) {
		fprintf(stderr, "Error: binary was not linked with libcudart. Relink code.\n");
		exit(1);
	}
}


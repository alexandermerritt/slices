/**
 * @file bypass.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 10, 2012
 * @brief Allow interposer to fall through to true CUDA implementation.
 */

#define _GNU_SOURCE
#include <dlfcn.h>

#include <assert.h>
#include <cuda_runtime_api.h>

extern struct bypass bypass;

/** Function pointers to the real library. */
struct bypass
{
	cudaError_t (*cudaBindTexture)(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t);
	cudaError_t (*cudaBindTextureToArray)(const struct textureReference*, const struct cudaArray*,const struct cudaChannelFormatDesc*);
	cudaError_t (*cudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t, cudaStream_t);
	cudaError_t (*cudaDriverGetVersion)(int*);
	cudaError_t (*cudaFreeArray)(struct cudaArray*);
	cudaError_t (*cudaFreeHost)(void*);
	cudaError_t (*cudaFree)(void*);
	cudaError_t (*cudaFuncGetAttributes)(struct cudaFuncAttributes*, const char*);
	cudaError_t (*cudaGetDeviceCount)(int*);
	cudaError_t (*cudaGetDevice)(int*);
	cudaError_t (*cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
	cudaError_t (*cudaGetLastError)(void);
	cudaError_t (*cudaGetTextureReference)(const struct textureReference**, const char*);
	cudaError_t (*cudaHostAlloc)(void**, size_t, unsigned int);
	cudaError_t (*cudaLaunch)(const char*);
	cudaError_t (*cudaMallocArray)(struct cudaArray**, const struct cudaChannelFormatDesc*, size_t, size_t, unsigned int);
	cudaError_t (*cudaMallocPitch)(void**, size_t*, size_t, size_t);
	cudaError_t (*cudaMalloc)(void**, size_t);
	cudaError_t (*cudaMemcpyAsync)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
	cudaError_t (*cudaMemcpyFromSymbol)(void*, const char*, size_t, size_t, enum cudaMemcpyKind);
	cudaError_t (*cudaMemcpyToArray)(struct cudaArray*, size_t, size_t, const void*, size_t, enum cudaMemcpyKind);
	cudaError_t (*cudaMemcpyToSymbolAsync)(const char*, const void*, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
	cudaError_t (*cudaMemcpyToSymbol)(const char*, const void*, size_t, size_t, enum cudaMemcpyKind);
	cudaError_t (*cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind);
	cudaError_t (*cudaMemGetInfo)(size_t*, size_t*);
	cudaError_t (*cudaMemset)(void*, int, size_t);
	cudaError_t (*cudaRuntimeGetVersion)(int*);
	cudaError_t (*cudaSetDeviceFlags)(unsigned int);
	cudaError_t (*cudaSetDevice)(int);
	cudaError_t (*cudaSetupArgument)(const void*, size_t, size_t);
	cudaError_t (*cudaSetValidDevices)(int*, int);
	cudaError_t (*cudaStreamCreate)(cudaStream_t*);
	cudaError_t (*cudaStreamDestroy)(cudaStream_t);
	cudaError_t (*cudaStreamQuery)(cudaStream_t);
	cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
	cudaError_t (*cudaThreadExit)(void);
	cudaError_t (*cudaThreadSynchronize)(void);

	void**	(*__cudaRegisterFatBinary)(void*);
	void	(*__cudaRegisterFunction)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
	void	(*__cudaRegisterTexture)(void**, const struct textureReference*, const void**, const char*, int, int, int);
	void	(*__cudaRegisterVar)(void**, char*, char*, const char*, int, int, int, int);
	void	(*__cudaUnregisterFatBinary)(void** fatCubinHandle);

	const char* (*cudaGetErrorString)(cudaError_t);
	struct cudaChannelFormatDesc (*cudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind);
};

void fill_bypass(struct bypass *bypass);

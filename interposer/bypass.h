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

/*
 * Function pointer typedefs
 */

typedef cudaError_t (*fn_cudaBindTexture)(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t);
typedef cudaError_t (*fn_cudaBindTextureToArray)(const struct textureReference*, const struct cudaArray*,const struct cudaChannelFormatDesc*);
typedef cudaError_t (*fn_cudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t, cudaStream_t);
typedef cudaError_t (*fn_cudaDriverGetVersion)(int*);
typedef cudaError_t (*fn_cudaFreeArray)(struct cudaArray*);
typedef cudaError_t (*fn_cudaFreeHost)(void*);
typedef cudaError_t (*fn_cudaFree)(void*);
typedef cudaError_t (*fn_cudaFuncGetAttributes)(struct cudaFuncAttributes*, const char*);
typedef cudaError_t (*fn_cudaGetDeviceCount)(int*);
typedef cudaError_t (*fn_cudaGetDevice)(int*);
typedef cudaError_t (*fn_cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
typedef cudaError_t (*fn_cudaGetLastError)(void);
typedef cudaError_t (*fn_cudaGetTextureReference)(const struct textureReference**, const char*);
typedef cudaError_t (*fn_cudaHostAlloc)(void**, size_t, unsigned int);
typedef cudaError_t (*fn_cudaLaunch)(const char*);
typedef cudaError_t (*fn_cudaMallocArray)(struct cudaArray**, const struct cudaChannelFormatDesc*, size_t, size_t, unsigned int);
typedef cudaError_t (*fn_cudaMallocPitch)(void**, size_t*, size_t, size_t);
typedef cudaError_t (*fn_cudaMalloc)(void**, size_t);
typedef cudaError_t (*fn_cudaMemcpyAsync)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*fn_cudaMemcpyFromSymbol)(void*, const char*, size_t, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*fn_cudaMemcpyToArray)(struct cudaArray*, size_t, size_t, const void*, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*fn_cudaMemcpyToSymbolAsync)(const char*, const void*, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*fn_cudaMemcpyToSymbol)(const char*, const void*, size_t, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*fn_cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*fn_cudaMemGetInfo)(size_t*, size_t*);
typedef cudaError_t (*fn_cudaMemset)(void*, int, size_t);
typedef cudaError_t (*fn_cudaRuntimeGetVersion)(int*);
typedef cudaError_t (*fn_cudaSetDeviceFlags)(unsigned int);
typedef cudaError_t (*fn_cudaSetDevice)(int);
typedef cudaError_t (*fn_cudaSetupArgument)(const void*, size_t, size_t);
typedef cudaError_t (*fn_cudaSetValidDevices)(int*, int);
typedef cudaError_t (*fn_cudaStreamCreate)(cudaStream_t*);
typedef cudaError_t (*fn_cudaStreamDestroy)(cudaStream_t);
typedef cudaError_t (*fn_cudaStreamQuery)(cudaStream_t);
typedef cudaError_t (*fn_cudaStreamSynchronize)(cudaStream_t);
typedef cudaError_t (*fn_cudaThreadExit)(void);
typedef cudaError_t (*fn_cudaThreadSynchronize)(void);

typedef void**	(*fn__cudaRegisterFatBinary)(void*);
typedef void	(*fn__cudaRegisterFunction)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
typedef void	(*fn__cudaRegisterTexture)(void**, const struct textureReference*, const void**, const char*, int, int, int);
typedef void	(*fn__cudaRegisterVar)(void**, char*, char*, const char*, int, int, int, int);
typedef void	(*fn__cudaUnregisterFatBinary)(void** fatCubinHandle);

typedef const char* (*fn_cudaGetErrorString)(cudaError_t);
typedef struct cudaChannelFormatDesc (*fn_cudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind);

/** Function pointers to the real library. */
struct bypass
{
	fn_cudaBindTexture         	cudaBindTexture;
	fn_cudaBindTextureToArray  	cudaBindTextureToArray;
	fn_cudaConfigureCall       	cudaConfigureCall;
	fn_cudaCreateChannelDesc   	cudaCreateChannelDesc;
	fn_cudaDriverGetVersion    	cudaDriverGetVersion;
	fn_cudaFreeArray           	cudaFreeArray;
	fn_cudaFree                	cudaFree;
	fn_cudaFreeHost            	cudaFreeHost;
	fn_cudaFuncGetAttributes   	cudaFuncGetAttributes;
	fn_cudaGetDeviceCount      	cudaGetDeviceCount;
	fn_cudaGetDevice           	cudaGetDevice;
	fn_cudaGetDeviceProperties 	cudaGetDeviceProperties;
	fn_cudaGetErrorString      	cudaGetErrorString;
	fn_cudaGetLastError        	cudaGetLastError;
	fn_cudaGetTextureReference 	cudaGetTextureReference;
	fn_cudaHostAlloc           	cudaHostAlloc;
	fn_cudaLaunch              	cudaLaunch;
	fn_cudaMallocArray         	cudaMallocArray;
	fn_cudaMalloc              	cudaMalloc;
	fn_cudaMallocPitch         	cudaMallocPitch;
	fn_cudaMemcpyAsync         	cudaMemcpyAsync;
	fn_cudaMemcpy              	cudaMemcpy;
	fn_cudaMemcpyFromSymbol    	cudaMemcpyFromSymbol;
	fn_cudaMemcpyToArray       	cudaMemcpyToArray;
	fn_cudaMemcpyToSymbolAsync 	cudaMemcpyToSymbolAsync;
	fn_cudaMemcpyToSymbol      	cudaMemcpyToSymbol;
	fn_cudaMemGetInfo          	cudaMemGetInfo;
	fn_cudaMemset              	cudaMemset;
	fn__cudaRegisterFatBinary  	__cudaRegisterFatBinary;
	fn__cudaRegisterFunction   	__cudaRegisterFunction;
	fn__cudaRegisterTexture    	__cudaRegisterTexture;
	fn__cudaRegisterVar        	__cudaRegisterVar;
	fn_cudaRuntimeGetVersion   	cudaRuntimeGetVersion;
	fn_cudaSetDevice           	cudaSetDevice;
	fn_cudaSetDeviceFlags      	cudaSetDeviceFlags;
	fn_cudaSetupArgument       	cudaSetupArgument;
	fn_cudaSetValidDevices     	cudaSetValidDevices;
	fn_cudaStreamCreate        	cudaStreamCreate;
	fn_cudaStreamDestroy       	cudaStreamDestroy;
	fn_cudaStreamQuery         	cudaStreamQuery;
	fn_cudaStreamSynchronize   	cudaStreamSynchronize;
	fn_cudaThreadExit          	cudaThreadExit;
	fn_cudaThreadSynchronize   	cudaThreadSynchronize;
	fn__cudaUnregisterFatBinary	__cudaUnregisterFatBinary;
};

void fill_bypass(struct bypass *bypass);

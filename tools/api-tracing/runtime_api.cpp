/**
 * @file runtime_api.cpp
 *
 * @date March 19, 2012
 * @author Alex Merritt, merritt.alex@gatech.edu
 *
 * @brief Intercept CUDA Runtime API calls to record an applications "CUDA
 * trace": sequence of calls made, frequency, data sizes, etc.
 */

/*-------------------------------------- INCLUDES ----------------------------*/

// System includes
#include <assert.h>
#include <dlfcn.h>
#include <stdbool.h>

// CUDA includes
#include <cuda_runtime_api.h>

// Directory-immediate includes
#include "trace.h"

/*-------------------------------------- INTERNAL STRUCTURES------------------*/

struct nv_api
{
	const char* (*cudaGetErrorString)(cudaError_t);
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
	cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
	cudaError_t (*cudaThreadExit)(void);
	cudaError_t (*cudaThreadSynchronize)(void);
	struct cudaChannelFormatDesc (*cudaCreateChannelDesc)(int x, int y, int z, int w, enum cudaChannelFormatKind);
	void**	(*__cudaRegisterFatBinary)(void*);
	void	(*__cudaRegisterFunction)(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
	void	(*__cudaRegisterTexture)(void**, const struct textureReference*, const void**, const char*, int, int, int);
	void	(*__cudaRegisterVar)(void**, char*, char*, const char*, int, int, int, int);
	void	(*__cudaUnregisterFatBinary)(void** fatCubinHandle);
};

static struct nv_api nv_api;
static bool ops_filled = false;
static unsigned int cubins = 0; /** number of registerFatBinary calls */

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

static void fill_ops(struct nv_api *ops)
{
	// these statements generate a warning each, unfortunately
	nv_api.cudaGetErrorString = dlsym(RTLD_NEXT, "cudaGetErrorString");
	nv_api.cudaBindTexture = dlsym(RTLD_NEXT, "cudaBindTexture");
	nv_api.cudaBindTextureToArray = dlsym(RTLD_NEXT, "cudaBindTextureToArray");
	nv_api.cudaConfigureCall = dlsym(RTLD_NEXT, "cudaConfigureCall");
	nv_api.cudaDriverGetVersion = dlsym(RTLD_NEXT, "cudaDriverGetVersion");
	nv_api.cudaFreeArray = dlsym(RTLD_NEXT, "cudaFreeArray");
	nv_api.cudaFreeHost = dlsym(RTLD_NEXT, "cudaFreeHost");
	nv_api.cudaFree = dlsym(RTLD_NEXT, "cudaFree");
	nv_api.cudaFuncGetAttributes = dlsym(RTLD_NEXT, "cudaFuncGetAttributes");
	nv_api.cudaGetDeviceCount = dlsym(RTLD_NEXT, "cudaGetDeviceCount");
	nv_api.cudaGetDevice = dlsym(RTLD_NEXT, "cudaGetDevice");
	nv_api.cudaGetDeviceProperties = dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
	nv_api.cudaGetLastError = dlsym(RTLD_NEXT, "cudaGetLastError");
	nv_api.cudaGetTextureReference = dlsym(RTLD_NEXT, "cudaGetTextureReference");
	nv_api.cudaHostAlloc = dlsym(RTLD_NEXT, "cudaHostAlloc");
	nv_api.cudaLaunch = dlsym(RTLD_NEXT, "cudaLaunch");
	nv_api.cudaMallocArray = dlsym(RTLD_NEXT, "cudaMallocArray");
	nv_api.cudaMallocPitch = dlsym(RTLD_NEXT, "cudaMallocPitch");
	nv_api.cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");

	nv_api.cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
	nv_api.cudaMemcpyAsync = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
	nv_api.cudaMemcpyToSymbol = dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
	nv_api.cudaMemcpyFromSymbol = dlsym(RTLD_NEXT, "cudaMemcpyFromSymbol");
	nv_api.cudaMemcpyToSymbolAsync = dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");
	nv_api.cudaMemcpyToArray = dlsym(RTLD_NEXT, "cudaMemcpyToArray");

	nv_api.cudaMemGetInfo = dlsym(RTLD_NEXT, "cudaMemGetInfo");
	nv_api.cudaMemset = dlsym(RTLD_NEXT, "cudaMemset");
	nv_api.cudaRuntimeGetVersion = dlsym(RTLD_NEXT, "cudaRuntimeGetVersion");
	nv_api.cudaSetDeviceFlags = dlsym(RTLD_NEXT, "cudaSetDeviceFlags");
	nv_api.cudaSetDevice = dlsym(RTLD_NEXT, "cudaSetDevice");
	nv_api.cudaSetupArgument = dlsym(RTLD_NEXT, "cudaSetupArgument");
	nv_api.cudaSetValidDevices = dlsym(RTLD_NEXT, "cudaSetValidDevices");
	nv_api.cudaStreamCreate = dlsym(RTLD_NEXT, "cudaStreamCreate");
	nv_api.cudaStreamSynchronize = dlsym(RTLD_NEXT, "cudaStreamSynchronize");
	nv_api.cudaThreadExit = dlsym(RTLD_NEXT, "cudaThreadExit");
	nv_api.cudaThreadSynchronize = dlsym(RTLD_NEXT, "cudaThreadSynchronize");
	nv_api.cudaCreateChannelDesc = dlsym(RTLD_NEXT, "cudaCreateChannelDesc");
	nv_api.__cudaRegisterFatBinary = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
	nv_api.__cudaRegisterFunction = dlsym(RTLD_NEXT, "__cudaRegisterFunction");
	nv_api.__cudaRegisterTexture = dlsym(RTLD_NEXT, "__cudaRegisterTexture");
	nv_api.__cudaRegisterVar = dlsym(RTLD_NEXT, "__cudaRegisterVar");
	nv_api.__cudaUnregisterFatBinary = dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");

	assert(nv_api.__cudaRegisterFatBinary); // sanity check

	trace_init();
}

/*-------------------------------------- INTERPOSING API ---------------------*/

//
// Thread Management API
//

cudaError_t cudaThreadExit(void)
{
	TRACE_PREOPS;
	return nv_api.cudaThreadExit();
}

cudaError_t cudaThreadSynchronize(void)
{
	TRACE_PREOPS;
	return nv_api.cudaThreadSynchronize();
}

//
// Error Handling API
//

const char* cudaGetErrorString(cudaError_t error)
{
	TRACE_PREOPS;
	return nv_api.cudaGetErrorString(error);
}

cudaError_t cudaGetLastError(void)
{
	TRACE_PREOPS;
	return nv_api.cudaGetLastError();
}

//
// Device Managment API
//

cudaError_t cudaGetDevice(int *device)
{
	TRACE_PREOPS;
	return nv_api.cudaGetDevice(device);
}

cudaError_t cudaGetDeviceCount(int *count)
{
	TRACE_PREOPS;
	return nv_api.cudaGetDeviceCount(count);
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	TRACE_PREOPS;
	return nv_api.cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaSetDevice(int device)
{
	TRACE_PREOPS;
	return nv_api.cudaSetDevice(device);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	TRACE_PREOPS;
	return nv_api.cudaSetDeviceFlags(flags);
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
	TRACE_PREOPS;
	return nv_api.cudaSetValidDevices(device_arr, len);
}

//
// Stream Management API
//

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	TRACE_PREOPS;
	return nv_api.cudaStreamCreate(pStream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	TRACE_PREOPS;
	return nv_api.cudaStreamSynchronize(stream);
}

//
// Execution Control API
//

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, cudaStream_t stream) {

	TRACE_PREOPS;
	return nv_api.cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
	TRACE_PREOPS;
	return nv_api.cudaFuncGetAttributes(attr, func);
}

cudaError_t cudaLaunch(const char *entry)
{
	TRACE_PREOPS;
	return nv_api.cudaLaunch(entry);
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	TRACE_PREOPS;
	return nv_api.cudaSetupArgument(arg, size, offset);
}

//
// Memory Management API
//

cudaError_t cudaFree(void * devPtr)
{
	TRACE_PREOPS;
	return nv_api.cudaFree(devPtr);
}

cudaError_t cudaFreeArray(struct cudaArray * array)
{
	TRACE_PREOPS;
	return nv_api.cudaFreeArray(array);
}

cudaError_t cudaFreeHost(void * ptr)
{
	TRACE_PREOPS;
	return nv_api.cudaFreeHost(ptr);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	TRACE_PREOPS;
	return nv_api.cudaHostAlloc(pHost, size, flags);
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	TRACE_PREOPS;
	return nv_api.cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocArray(
		struct cudaArray **array, // stores a device address
		const struct cudaChannelFormatDesc *desc,
		size_t width, size_t height, unsigned int flags)
{
	TRACE_PREOPS;
	return nv_api.cudaMallocArray(array, desc, width, height, flags);
}

cudaError_t cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height)
{
	TRACE_PREOPS;
	return nv_api.cudaMallocPitch(devPtr, pitch, width, height);
}

cudaError_t cudaMemcpy(void *dst, const void *src,
		size_t count, enum cudaMemcpyKind kind)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaMemcpyFromSymbol(
		void *dst,
		const char *symbol, //! Either an addr of a var in the app, or a string
		size_t count, size_t offset,
		enum cudaMemcpyKind kind)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
}

cudaError_t cudaMemcpyToArray(
		struct cudaArray *dst,
		size_t wOffset, size_t hOffset,
		const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}


cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset,
		enum cudaMemcpyKind kind)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyToSymbolAsync(
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	TRACE_PREOPS;
	return nv_api.cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	TRACE_PREOPS;
	return nv_api.cudaMemGetInfo(free, total);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	TRACE_PREOPS;
	return nv_api.cudaMemset(devPtr, value, count);
}

//
// Texture Management API
//

// see comments in __cudaRegisterTexture and cudaBindTextureToArray
cudaError_t cudaBindTexture(size_t *offset,
		const struct textureReference *texRef, //! addr of global var in app
		const void *devPtr,
		const struct cudaChannelFormatDesc *desc,
		size_t size)
{
	TRACE_PREOPS;
	return nv_api.cudaBindTexture(offset, texRef, devPtr, desc, size);
}

cudaError_t cudaBindTextureToArray(
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	TRACE_PREOPS;
	return nv_api.cudaBindTextureToArray(texRef, array, desc);
}

struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
		enum cudaChannelFormatKind format)
{
	TRACE_PREOPS;
	return nv_api.cudaCreateChannelDesc(x, y, z, w, format);
}

cudaError_t cudaGetTextureReference(
		const struct textureReference **texRef,
		const char *symbol)
{
	TRACE_PREOPS;
	return nv_api.cudaGetTextureReference(texRef, symbol);
}

//
// Version Management API
//

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	TRACE_PREOPS;
	return nv_api.cudaDriverGetVersion(driverVersion);
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	TRACE_PREOPS;
	return nv_api.cudaRuntimeGetVersion(runtimeVersion);
}

//
// Undocumented API
// 	extern "C" is required when compiling with g++ else the function names are
// 	modified.. meaning they do not get interposed
//

extern "C"
void** __cudaRegisterFatBinary(void* cubin)
{
	if (!ops_filled) {
		fill_ops(&nv_api);
		ops_filled = true;
	}
	cubins++;
	TRACE_PREOPS;
	return nv_api.__cudaRegisterFatBinary(cubin);
}

extern "C"
void __cudaUnregisterFatBinary(void** fatCubinHandle)
{
	TRACE_PREOPS;
	cubins--;
	nv_api.__cudaUnregisterFatBinary(fatCubinHandle);
	if (cubins <= 0)
		trace_report();
}

extern "C"
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
	TRACE_PREOPS;
	nv_api.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
			deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

extern "C"
void __cudaRegisterVar(
		void **fatCubinHandle,	//! cubin this var associates with
		char *hostVar,			//! addr of a var within app (not string)
		char *deviceAddress,	//! 8-byte device addr
		const char *deviceName, //! actual string
		int ext, int vsize, int constant, int global)
{
	TRACE_PREOPS;
	return nv_api.__cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
			deviceName, ext, vsize, constant, global);
}

// first three args we treat as handles (i.e. only pass down the pointer
// addresses)
extern "C"
void __cudaRegisterTexture(
		void** fatCubinHandle,
		//! address of a global variable within the application; store the addr
		const struct textureReference* texRef,
		//! 8-byte device address; dereference it once to get the address
		const void** deviceAddress,
		const char* texName, //! actual string
		int dim, int norm, int ext)
{
	TRACE_PREOPS;
	return nv_api.__cudaRegisterTexture(fatCubinHandle, texRef, deviceAddress,
			texName, dim, norm, ext);
}

/*-------------------------------------- ^^ CODE TO MERGE UP ^^ --------------*/

#if 0
cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value)

cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit)

cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig)

cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig)

cudaError_t cudaPeekAtLastError(void)

cudaError_t cudaStreamDestroy(cudaStream_t stream)

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)

cudaError_t cudaStreamQuery(cudaStream_t stream)

cudaError_t cudaEventCreate(cudaEvent_t *event)

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0))

cudaError_t cudaEventQuery(cudaEvent_t event)

cudaError_t cudaEventSynchronize(cudaEvent_t event)

cudaError_t cudaEventDestroy(cudaEvent_t event) 

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) 

cudaError_t cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache cacheConfig) 

cudaError_t cudaSetDoubleForDevice(double *d) 

cudaError_t cudaSetDoubleForHost(double *d) 

cudaError_t cudaMallocHost(void **ptr, size_t size) 

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost)

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent)

cudaError_t cudaMalloc3DArray(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags)

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p)

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)

cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) 

cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)

cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0))

cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0))

cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0))

cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol)

cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol)

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource)

cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned int flags)

cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0))

cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0))

cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource)

cudaError_t cudaGraphicsSubResourceGetMappedArray( struct cudaArray **array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel)

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)

cudaError_t cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch)

cudaError_t cudaBindTextureToArray( const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)

cudaError_t cudaUnbindTexture(const struct textureReference *texref)

cudaError_t cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)

cudaError_t cudaBindSurfaceToArray( const struct surfaceReference *surfref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)

cudaError_t cudaGetSurfaceReference( const struct surfaceReference **surfref, const char *symbol)

cudaError_t cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId)

void __cudaRegisterShared(void** fatCubinHandle, void** devicePtr)
#endif

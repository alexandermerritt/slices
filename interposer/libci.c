/**
 * @file libci.c
 *
 * @date Feb 27, 2011
 * @author Magda Slawinska, magg@gatech.edu

 * @brief Interposes the cuda calls and prints arguments of the call. It supports
 * the 3.2 CUDA Toolkit, specifically
 *   CUDA Runtime API Version 3.2
 *   #define CUDART_VERSION  3020
 *
 * To prepare the file I processed the /opt/cuda/include/cuda_runtime_api_no_comments.h
 * and removed the comments. I also removed CUDARTAPI and __host__ modifiers.
 * Then I have a list of function signatures that I need to interpose.
 * You can see the script in cuda_rt_api.
 *
 * 2011-02-09 It looks that  in my library currently is  95 calls plus 6 calls undocumented.
 * @todo Write a script that checks if the number or api are identical in cuda_runtime_api.h
 * and in our file.
 *
 *
 * @todo There is one thing: the prototypes or signatures of CUDA functions
 * have modifiers CUDARTAPI which is __stdcall and __host__ which is
 * __location__(host) as defined in file /opt/cuda/include/host_defines.h
 * The question is if this has any impact on the interposed calls. I guess not.
 * But I might be wrong.
 *
 * To use that library you can do
 * [magg@prost release]$ LD_PRELOAD=/home/magg/libs/libci.so ./deviceQuery
 *
 * The library interposes calls and delegates the execution to the other machine
 */

// if this file is .c, if you do not define _GNU_SOURCE then it complains
// that RTLD_NEXT is undeclared
// if this file is  .cpp, then you get warning "_GNU_SOURCE" redefined
#define _GNU_SOURCE

#include <stdio.h>
#include <driver_types.h>

// /opt/cuda/include for uint3
#include <vector_types.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include <cuda.h>		// for CUDA_SUCCESS
#include <__cudaFatFormat.h>  // for __cudaFatCudaBinary
#include "packetheader.h" 	// for cuda_packet_t
#include "debug.h"			// printd, ERROR, OK
#include <pthread.h>		// pthread_self()
#include "method_id.h"		// method identifiers
#include "remote_api_wrapper.h" // for nvback....rpc/srv functions
#include "connection.h"
#include "libciutils.h"
#include "local_api_wrapper.h"  // for copyFatBinary

//! to indicate the error with the dynamic loaded library
static cudaError_t cudaErrorDL = cudaErrorUnknown;

//! Maintain the last error for cudaGetLastError()
static cudaError_t cuda_err = 0;

//! Right now the host where we are connecting to
const char * SERVER_HOSTNAME = "cuda2.cc.gt.atl.ga.us";



/**
 * @brief Handles errors caused by dlsym()
 * @return true no error - everything ok, otherwise the false
 */
int l_handleDlError() {
	char * error; // handles error description
	int ret = OK; // return value

	if ((error = dlerror()) != NULL) {
		printf("%s.%d: %s\n", __FUNCTION__, __LINE__, error);
		ret = ERROR;
	}

	return ret;
}
/**
 * Prints function signature
 * @param pSignature The string describing the function signature
 * @return always true
 */
int l_printFuncSig(const char* pSignature) {
	printf(">>>>>>>>>> %s\n", pSignature);
	//std::cout << ">>>>>>>>>> " << pSignature << std::endl;
	return OK;
}
/**
 * Prints function signature; should be used for the
 * implemented functions
 * @param pSignature The string describing the function signature
 * @return always true
 */
int l_printFuncSigImpl(const char* pSignature) {
	printf(">>>>>>>>>> Implemented: %s\n", pSignature);
	//std::cout << ">>>>>>>>>> " << pSignature << std::endl;
	return OK;
}
/**
 * sets the method_id, thr_id, flags in the packet structure to default values
 * @param pPacket The packet to be changed
 * @param methodId The method id you want to set
 *
 */
int l_setMetThrReq(cuda_packet_t ** const pPacket, const uint16_t methodId){
	(*pPacket)->method_id = methodId;
	(*pPacket)->thr_id = pthread_self();
	(*pPacket)->flags = CUDA_request;

	return OK;
}



//! This appears in some values of arguments. I took this from /opt/cuda/include/cuda_runtime_api.h
//! It looks as this comes from a default value (dv)
#if !defined(__dv)
#	if defined(__cplusplus)
#		define __dv(v) \
        		= v
#	else
#		define __dv(v)
#	endif
#endif


cudaError_t cudaThreadExit(void) {
	typedef cudaError_t (* pFuncType)(void);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadExit");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc());
}
cudaError_t cudaThreadSynchronize(void) {
	typedef cudaError_t (* pFuncType)(void);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadSynchronize");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc());
}
cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value) {
	typedef cudaError_t (* pFuncType)(enum cudaLimit limit, size_t value);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadSetLimit");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(limit, value));
}
cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit) {
	typedef cudaError_t (* pFuncType)(size_t *pValue, enum cudaLimit limit);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadGetLimit");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pValue, limit));
}
cudaError_t cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
	typedef cudaError_t (* pFuncType)(enum cudaFuncCache *pCacheConfig);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadGetCacheConfig");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pCacheConfig));
}
cudaError_t cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
	typedef cudaError_t (* pFuncType)(enum cudaFuncCache cacheConfig);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaThreadSetCacheConfig");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(cacheConfig));
}
// -------------------------------------------
cudaError_t cudaGetLastError(void) {
	typedef cudaError_t (* pFuncType)(void);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetLastError");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc());
}
cudaError_t cudaPeekAtLastError(void) {
	typedef cudaError_t (* pFuncType)(void);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaPeekAtLastError");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc());
}
const char* cudaGetErrorString(cudaError_t error) {
	typedef const char* (* pFuncType)(cudaError_t error);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetErrorString");

		if (l_handleDlError() != 0)
			return "DL error";
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(error));
}
// ----------------------------------
cudaError_t cudaGetDeviceCount(int *count) {
	cuda_packet_t * pPacket;

	l_printFuncSigImpl(__FUNCTION__);

	// Now make a packet and send
	if ((pPacket = callocCudaPacket(__FUNCTION__, &cuda_err)) == NULL) {
		return cuda_err;
	}

	l_setMetThrReq(&pPacket, CUDA_GET_DEVICE_COUNT);

	// send the packet
	if(nvbackCudaGetDeviceCount_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s.%d: Return from rpc with the wrong return value.\n", __FUNCTION__, __LINE__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		printd(DBG_INFO, "%s.%d: the number of devices is %ld. Got from the RPC call\n", __FUNCTION__, __LINE__,
				pPacket->args[0].argi);
		// remember the count number what we get from the remote device
		*count = pPacket->args[0].argi;
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);


	// TODO call the function locally right now - it should be removed in the future
	// when it should react appropriately and call rpc or local version.


	/*
	typedef cudaError_t (* pFuncType)(int *count);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetDeviceCount");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	return (pFunc(count)); */
	return cuda_err;
}

/**
 * Prints the device properties
 */
int l_printCudaDeviceProp(const struct cudaDeviceProp * const pProp){
	printd(DBG_INFO, "\nDevice \"%s\"\n",  pProp->name);
	printd(DBG_INFO, "  CUDA Capability Major/Minor version number:    %d.%d\n", pProp->major, pProp->minor);
	printd(DBG_INFO, "  Total amount of global memory:                 %llu bytes\n", (unsigned long long) pProp->totalGlobalMem);
	printd(DBG_INFO, "  Multiprocessors: %d (MP) \n", pProp->multiProcessorCount);
    printd(DBG_INFO, "  Total amount of constant memory:               %lu bytes\n", pProp->totalConstMem);
    printd(DBG_INFO, "  Total amount of shared memory per block:       %lu bytes\n", pProp->sharedMemPerBlock);
    printd(DBG_INFO, "  Total number of registers available per block: %d\n", pProp->regsPerBlock);
    printd(DBG_INFO, "  Warp size:                                     %d\n", pProp->warpSize);
    printd(DBG_INFO, "  Maximum number of threads per block:           %d\n", pProp->maxThreadsPerBlock);
    printd(DBG_INFO, "  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
           pProp->maxThreadsDim[0],
           pProp->maxThreadsDim[1],
           pProp->maxThreadsDim[2]);
    printd(DBG_INFO, "  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
           pProp->maxGridSize[0],
           pProp->maxGridSize[1],
           pProp->maxGridSize[2]);
    return OK;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
	cuda_packet_t *pPacket;

	l_printFuncSigImpl(__FUNCTION__);
	// Now make a packet and send
	if ((pPacket = callocCudaPacket(__FUNCTION__, &cuda_err)) == NULL) {
		return cuda_err;
	}

	l_setMetThrReq(&pPacket, CUDA_GET_DEVICE_PROPERTIES);
	// override the flags
	pPacket->flags = CUDA_request | CUDA_Copytype;
	pPacket->args[0].argp = (void *) prop; // I do not understand why we do this
	pPacket->args[1].argi = device;   // I understand this
	pPacket->args[2].argi = sizeof(struct cudaDeviceProp); // for driver; I do not understand why we do this

	// send the packet
	if (nvbackCudaGetDeviceProperties_rpc(pPacket) != OK) {
		printd(DBG_ERROR, "%s.%d: Return from rpc with the wrong return value.\n", __FUNCTION__, __LINE__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		l_printCudaDeviceProp(prop);
		// remember the count number what we get from the remote device
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);

	return cuda_err;


/*	typedef cudaError_t (* pFuncType)(struct cudaDeviceProp *prop, int device);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetDeviceProperties");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(prop, device));
	*/
}
cudaError_t cudaChooseDevice(int *device,
		const struct cudaDeviceProp *prop) {
	typedef cudaError_t (* pFuncType)(int *device,
			const struct cudaDeviceProp *prop);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaChooseDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device, prop));
}
cudaError_t cudaSetDevice(int device) {
	typedef cudaError_t (* pFuncType)(int device);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device));
}
cudaError_t cudaGetDevice(int *device) {
	typedef cudaError_t (* pFuncType)(int *device);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device));
}
cudaError_t cudaSetValidDevices(int *device_arr, int len) {
	typedef cudaError_t (* pFuncType)(int *device_arr, int len);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetValidDevices");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device_arr, len));
}
cudaError_t cudaSetDeviceFlags(unsigned int flags) {
	typedef cudaError_t (* pFuncType)(unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetDeviceFlags");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(flags));
}
// -----------------------------------------
cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
	typedef cudaError_t (* pFuncType)(cudaStream_t *pStream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaStreamCreate");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pStream));
}
cudaError_t cudaStreamDestroy(cudaStream_t stream) {
	typedef cudaError_t (* pFuncType)(cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaStreamDestroy");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(stream));
}
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
		unsigned int flags) {
	typedef cudaError_t (* pFuncType)(cudaStream_t stream, cudaEvent_t event,
			unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaStreamWaitEvent");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(stream, event, flags));
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
	typedef cudaError_t (* pFuncType)(cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaStreamSynchronize");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(stream));
}
cudaError_t cudaStreamQuery(cudaStream_t stream) {
	typedef cudaError_t (* pFuncType)(cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaStreamQuery");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(stream));
}
// --------------------------------

cudaError_t cudaEventCreate(cudaEvent_t *event) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t *event);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventCreate");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event));
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
		unsigned int flags) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t *event, unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event, flags));
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream
		__dv(0)) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t event, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventRecord");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event, stream));
}
cudaError_t cudaEventQuery(cudaEvent_t event) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t event);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventQuery");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event));
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t event);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventSynchronize");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event));
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
	typedef cudaError_t (* pFuncType)(cudaEvent_t event);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventDestroy");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(event));
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
		cudaEvent_t end) {
	typedef cudaError_t (* pFuncType)(float *ms, cudaEvent_t start,
			cudaEvent_t end);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaEventElapsedTime");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(ms, start, end));
}

// --------------------------------------------
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem  __dv(0), cudaStream_t stream  __dv(0)) {
	typedef cudaError_t (* pFuncType)(dim3 gridDim, dim3 blockDim,
			size_t sharedMem, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaConfigureCall");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(gridDim, blockDim, sharedMem, stream));
}
cudaError_t cudaSetupArgument(const void *arg, size_t size,
		size_t offset) {
	typedef cudaError_t (* pFuncType)(const void *arg, size_t size,
			size_t offset);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetupArgument");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(arg, size, offset));
}
cudaError_t cudaFuncSetCacheConfig(const char *func,
		enum cudaFuncCache cacheConfig) {
	typedef cudaError_t (* pFuncType)(const char *func,
			enum cudaFuncCache cacheConfig);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaFuncSetCacheConfig");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(func, cacheConfig));
}
cudaError_t cudaLaunch(const char *entry) {
	typedef cudaError_t (* pFuncType)(const char *entry);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaLaunch");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(entry));
}
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
		const char *func) {
	typedef cudaError_t (* pFuncType)(struct cudaFuncAttributes *attr,
			const char *func);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaFuncGetAttributes");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(attr, func));
}
cudaError_t cudaSetDoubleForDevice(double *d) {
	typedef cudaError_t (* pFuncType)(double *d);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetDoubleForDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(d));
}
cudaError_t cudaSetDoubleForHost(double *d) {
	l_printFuncSig(__FUNCTION__);

	typedef cudaError_t (* pFuncType)(double *d);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetDoubleForHost");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	return (pFunc(d));
}
// --------------------------------------------
cudaError_t cudaMalloc(void **devPtr, size_t size) {
	cuda_packet_t * pPacket;

	l_printFuncSigImpl(__FUNCTION__);

	// Now make a packet and send
	if ((pPacket = callocCudaPacket(__FUNCTION__, &cuda_err)) == NULL) {
		return cuda_err;
	}

	l_setMetThrReq(&pPacket, CUDA_MALLOC);
	pPacket->args[0].argdp = devPtr;
	pPacket->args[1].argi = size;

	if(nvbackCudaMalloc_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s.%d: Return from the RPC with an error\n", __FUNCTION__, __LINE__);
		cuda_err = cudaErrorMemoryAllocation;
		*devPtr = NULL;
	} else {
		printd(DBG_INFO, "%s.%d: Return from the RPC call DevPtr and size: %p\n", __FUNCTION__, __LINE__,
				pPacket->args[0].argdp);
		// unpack what we have got from the packet
		*devPtr = pPacket->args[0].argp;
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);

	/*
	typedef cudaError_t (* pFuncType)(void **devPtr, size_t size);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMalloc");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	return (pFunc(devPtr, size)); */
	return cuda_err;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
	typedef cudaError_t (* pFuncType)(void **ptr, size_t size);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMallocHost");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(ptr, size));
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
		size_t height) {
	typedef cudaError_t (* pFuncType)(void **devPtr, size_t *pitch,
			size_t width, size_t height);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMallocPitch");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, pitch, width, height));
}

cudaError_t cudaMallocArray(struct cudaArray **array,
		const struct cudaChannelFormatDesc *desc, size_t width, size_t height
				__dv(0), unsigned int flags __dv(0)) {
	typedef cudaError_t (* pFuncType)(struct cudaArray **,
			const struct cudaChannelFormatDesc *, size_t, size_t, unsigned int);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMallocArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(array, desc, width, height, flags));
}

/**
 * @brief The pattern of the functions so far. This function is being interposed
 * via PRE_LOAD or something like that. We need the original function; that's
 * why we call dlsym that seeks to find the next call of this signature. So first
 * we find that call, store in a function pointer and later call it eventually.
 * Right now, the only thing that interposed function does prints the arguments
 * of this function.
 */
cudaError_t cudaFree(void * devPtr) {
	cuda_packet_t *pPacket;

	// print the function name and the parameters
	l_printFuncSigImpl(__FUNCTION__);

	// Now make a packet and send
	if ((pPacket = callocCudaPacket(__FUNCTION__, &cuda_err)) == NULL) {
		return cuda_err;
	}

	l_setMetThrReq(&pPacket, CUDA_FREE);
	pPacket->args[0].argp = devPtr;

	// send the packet
	if(nvbackCudaFree_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s.%d: Return from rpc with the wrong return value.\n", __FUNCTION__, __LINE__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		printd(DBG_INFO, "%s.%d: The used pointer %p\n", __FUNCTION__, __LINE__,
				pPacket->args[0].argp);
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);

	return cuda_err;

/*	typedef cudaError_t (* pFuncType)(void *);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		// find next occurence of the cudaFree() call,
		// C++ requires casting void to the func ptr
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaFree");

		if (l_handleDlError() != 0)
			// problems with the dynamic library
			return cudaErrorDL;
	}


	// call the function that was found by the dlsym - we hope this is
	// the original function
	return (pFunc(devPtr)); */
}

cudaError_t cudaFreeHost(void * ptr) {
	typedef cudaError_t (* pFuncType)(void *);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaFreeHost");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(ptr));
}

cudaError_t cudaFreeArray(struct cudaArray * array) {
	typedef cudaError_t (* pFuncType)(struct cudaArray * array);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaFreeArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(array));
}

// --------------------------------------------------------
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
	typedef cudaError_t (* pFuncType)(void **, size_t, unsigned int);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaHostAlloc");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pHost, size, flags));
}

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
		unsigned int flags) {
	typedef cudaError_t (* pFuncType)(void **pDevice, void *pHost,
			unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaHostGetDevicePointer");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(pDevice, pHost, flags));
}

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
	typedef cudaError_t (* pFuncType)(unsigned int*, void*);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaHostGetFlags");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pFlags, pHost));
}

cudaError_t cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr,
		struct cudaExtent extent) {
	typedef cudaError_t
	(* pFuncType)(struct cudaPitchedPtr*, struct cudaExtent);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMalloc3D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pitchedDevPtr, extent));
}

cudaError_t cudaMalloc3DArray(struct cudaArray** array,
		const struct cudaChannelFormatDesc* desc, struct cudaExtent extent,
		unsigned int flags) {
	typedef cudaError_t (* pFuncType)(struct cudaArray** array,
			const struct cudaChannelFormatDesc* desc, struct cudaExtent extent,
			unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMalloc3DArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(array, desc, extent, flags));
}
// --------------------------------
cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
	typedef cudaError_t (* pFuncType)(const struct cudaMemcpy3DParms *p);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy3D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(p));
}
cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(const struct cudaMemcpy3DParms *p,
			cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy3DAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(p, stream));
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
	typedef cudaError_t (* pFuncType)(size_t *free, size_t *total);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemGetInfo");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(free, total));
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind) {
	typedef cudaError_t (* pFuncType)(void *dst, const void *src, size_t count,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, src, count, kind));
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffset,
			size_t hOffset, const void *src, size_t count,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffset, hOffset, src, count, kind));
}

cudaError_t cudaMemcpyFromArray(void *dst, const struct cudaArray *src,
		size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
	typedef cudaError_t (* pFuncType)(void *dst, const struct cudaArray *src,
			size_t wOffset, size_t hOffset, size_t count,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyFromArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, src, wOffset, hOffset, count, kind));
}

cudaError_t cudaMemcpyArrayToArray(struct cudaArray *dst,
		size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
		size_t wOffsetSrc, size_t hOffsetSrc, size_t count,
		enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffsetDst,
			size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
			size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyArrayToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc,
			count, kind));

}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
		size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
	typedef cudaError_t (* pFuncType)(void *dst, size_t dpitch,
			const void *src, size_t spitch, size_t width, size_t height,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, dpitch, src, spitch, width, height, kind));
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t spitch, size_t width,
		size_t height, enum cudaMemcpyKind kind) {

	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffset,
			size_t hOffset, const void *src, size_t spitch, size_t width,
			size_t height, enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffset, hOffset, src, spitch, width, height, kind));
}

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
		const struct cudaArray *src, size_t wOffset, size_t hOffset,
		size_t width, size_t height, enum cudaMemcpyKind kind) {
	typedef cudaError_t (* pFuncType)(void *dst, size_t dpitch,
			const struct cudaArray *src, size_t wOffset, size_t hOffset,
			size_t width, size_t height, enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DFromArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, dpitch, src, wOffset, hOffset, width, height, kind));
}

cudaError_t cudaMemcpy2DArrayToArray(struct cudaArray *dst,
		size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src,
		size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
		enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffsetDst,
			size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc,
			size_t hOffsetSrc, size_t width, size_t height,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DArrayToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc,
			width, height, kind));
}

cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src,
		size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind
				__dv(cudaMemcpyHostToDevice)) {
	typedef cudaError_t (* pFuncType)(const char *symbol, const void *src,
			size_t count, size_t offset, enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(symbol, src, count, offset, kind));
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol,
		size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind
				__dv(cudaMemcpyDeviceToHost)) {
	typedef cudaError_t (* pFuncType)(void *dst, const char *symbol,
			size_t count, size_t offset, enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyFromSymbol");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, symbol, count, offset, kind));
}

// -----------------------------------
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *dst, const void *src, size_t count,
			enum cudaMemcpyKind kind, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, src, count, kind, stream));
}

cudaError_t cudaMemcpyToArrayAsync(struct cudaArray *dst,
		size_t wOffset, size_t hOffset, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffset,
			size_t hOffset, const void *src, size_t count,
			enum cudaMemcpyKind kind, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyToArrayAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffset, hOffset, src, count, kind, stream));
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst,
		const struct cudaArray *src, size_t wOffset, size_t hOffset,
		size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *dst, const struct cudaArray *src,
			size_t wOffset, size_t hOffset, size_t count,
			enum cudaMemcpyKind kind, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyFromArrayAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, src, wOffset, hOffset, count, kind, stream));
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
		size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *dst, size_t dpitch,
			const void *src, size_t spitch, size_t width, size_t height,
			enum cudaMemcpyKind kind, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, dpitch, src, spitch, width, height, kind, stream));
}

cudaError_t cudaMemcpy2DToArrayAsync(struct cudaArray *dst,
		size_t wOffset, size_t hOffset, const void *src, size_t spitch,
		size_t width, size_t height, enum cudaMemcpyKind kind,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(struct cudaArray *dst, size_t wOffset,
			size_t hOffset, const void *src, size_t spitch, size_t width,
			size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DToArrayAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, wOffset, hOffset, src, spitch, width, height, kind,
			stream));
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
		const struct cudaArray *src, size_t wOffset, size_t hOffset,
		size_t width, size_t height, enum cudaMemcpyKind kind,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *dst, size_t dpitch,
			const struct cudaArray *src, size_t wOffset, size_t hOffset,
			size_t width, size_t height, enum cudaMemcpyKind kind,
			cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy2DFromArrayAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, dpitch, src, wOffset, hOffset, width, height, kind,
			stream));
}

cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src,
		size_t count, size_t offset, enum cudaMemcpyKind kind,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(const char *symbol, const void *src,
			size_t count, size_t offset, enum cudaMemcpyKind kind,
			cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyToSymbolAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(symbol, src, count, offset, kind, stream));
}

cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol,
		size_t count, size_t offset, enum cudaMemcpyKind kind,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *dst, const char *symbol,
			size_t count, size_t offset, enum cudaMemcpyKind kind,
			cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpyFromSymbolAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, symbol, count, offset, kind, stream));
}

// -------------------------------------
cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
	typedef cudaError_t (* pFuncType)(void *devPtr, int value, size_t count);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemset");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, value, count));
}

cudaError_t cudaMemset2D(void *devPtr, size_t pitch, int value,
		size_t width, size_t height) {
	typedef cudaError_t (* pFuncType)(void *devPtr, size_t pitch, int value,
			size_t width, size_t height);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemset2D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, pitch, value, width, height));
}
cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value,
		struct cudaExtent extent) {
	typedef cudaError_t (* pFuncType)(struct cudaPitchedPtr pitchedDevPtr,
			int value, struct cudaExtent extent);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemset3D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pitchedDevPtr, value, extent));
}
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
		cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *devPtr, int value, size_t count,
			cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemsetAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, value, count, stream));
}
cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch, int value,
		size_t width, size_t height, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(void *devPtr, size_t pitch, int value,
			size_t width, size_t height, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemset2DAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, pitch, value, width, height, stream));
}
cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr,
		int value, struct cudaExtent extent, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(struct cudaPitchedPtr pitchedDevPtr,
			int value, struct cudaExtent extent, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemset3DAsync");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(pitchedDevPtr, value, extent, stream));
}

cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol) {
	typedef cudaError_t (* pFuncType)(void **devPtr, const char *symbol);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetSymbolAddress");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, symbol));
}

cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol) {
	typedef cudaError_t (* pFuncType)(size_t *size, const char *symbol);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetSymbolSize");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(size, symbol));
}

// -----------------------
cudaError_t cudaGraphicsUnregisterResource(
		cudaGraphicsResource_t resource) {
	typedef cudaError_t (* pFuncType)(cudaGraphicsResource_t resource);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGraphicsUnregisterResource");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(resource));
}
cudaError_t cudaGraphicsResourceSetMapFlags(
		cudaGraphicsResource_t resource, unsigned int flags) {
	typedef cudaError_t (* pFuncType)(cudaGraphicsResource_t resource,
			unsigned int flags);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGraphicsResourceSetMapFlags");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(resource, flags));
}
cudaError_t cudaGraphicsMapResources(int count,
		cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(int count,
			cudaGraphicsResource_t *resources, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGraphicsMapResources");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(count, resources, stream));
}
cudaError_t cudaGraphicsUnmapResources(int count,
		cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0)) {
	typedef cudaError_t (* pFuncType)(int count,
			cudaGraphicsResource_t *resources, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGraphicsUnmapResources");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(count, resources, stream));
}
cudaError_t cudaGraphicsResourceGetMappedPointer(void **devPtr,
		size_t *size, cudaGraphicsResource_t resource) {
	typedef cudaError_t (* pFuncType)(void **devPtr, size_t *size,
			cudaGraphicsResource_t resource);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT,
				"cudaGraphicsResourceGetMappedPointer");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(devPtr, size, resource));
}
cudaError_t cudaGraphicsSubResourceGetMappedArray(
		struct cudaArray **array, cudaGraphicsResource_t resource,
		unsigned int arrayIndex, unsigned int mipLevel) {
	typedef cudaError_t (* pFuncType)(struct cudaArray **array,
			cudaGraphicsResource_t resource, unsigned int arrayIndex,
			unsigned int mipLevel);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT,
				"cudaGraphicsSubResourceGetMappedArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(array, resource, arrayIndex, mipLevel));
}

// --------------------------
cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
		const struct cudaArray *array) {
	typedef cudaError_t (* pFuncType)(struct cudaChannelFormatDesc *desc,
			const struct cudaArray *array);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetChannelDesc");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(desc, array));

}
/**
 * This call returns something different than cudaError_t so we must use different something
 * else than cudaErrorDL
 * @todo better handle DL error
 * @return empty cudaChannelFormatDesc if there is a problem with DL, as well (maybe other NULL might means something else)
 */
struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z,
		int w, enum cudaChannelFormatKind f) {
	typedef struct cudaChannelFormatDesc (* pFuncType)(int x, int y, int z,
			int w, enum cudaChannelFormatKind f);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaCreateChannelDesc");

		if (l_handleDlError() != 0) {
			struct cudaChannelFormatDesc desc;
			return desc;
		}

	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(x, y, z, w, f));

}

// --------------------------
cudaError_t cudaBindTexture(size_t *offset,
		const struct textureReference *texref, const void *devPtr,
		const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) {
	typedef cudaError_t (* pFuncType)(size_t *offset,
			const struct textureReference *texref, const void *devPtr,
			const struct cudaChannelFormatDesc *desc, size_t size);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaBindTexture");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(offset, texref, devPtr, desc, size));

}
cudaError_t cudaBindTexture2D(size_t *offset,
		const struct textureReference *texref, const void *devPtr,
		const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
		size_t pitch) {
	typedef cudaError_t (* pFuncType)(size_t *offset,
			const struct textureReference *texref, const void *devPtr,
			const struct cudaChannelFormatDesc *desc, size_t width,
			size_t height, size_t pitch);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaBindTexture2D");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(offset, texref, devPtr, desc, width, height, pitch));
}
cudaError_t cudaBindTextureToArray(
		const struct textureReference *texref, const struct cudaArray *array,
		const struct cudaChannelFormatDesc *desc) {
	typedef cudaError_t (* pFuncType)(const struct textureReference *texref,
			const struct cudaArray *array,
			const struct cudaChannelFormatDesc *desc);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaBindTextureToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(texref, array, desc));

}
cudaError_t cudaUnbindTexture(const struct textureReference *texref) {
	typedef cudaError_t (* pFuncType)(const struct textureReference *texref);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaUnbindTexture");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(texref));

}
cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
		const struct textureReference *texref) {
	typedef cudaError_t (* pFuncType)(size_t *offset,
			const struct textureReference *texref);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetTextureAlignmentOffset");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(offset, texref));
}
cudaError_t cudaGetTextureReference(
		const struct textureReference **texref, const char *symbol) {
	typedef cudaError_t (* pFuncType)(const struct textureReference **texref,
			const char *symbol);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetTextureReference");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(texref, symbol));

}

// --------------------------
cudaError_t cudaBindSurfaceToArray(
		const struct surfaceReference *surfref, const struct cudaArray *array,
		const struct cudaChannelFormatDesc *desc) {
	typedef cudaError_t (* pFuncType)(const struct surfaceReference *surfref,
			const struct cudaArray *array,
			const struct cudaChannelFormatDesc *desc);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaBindSurfaceToArray");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(surfref, array, desc));

}
cudaError_t cudaGetSurfaceReference(
		const struct surfaceReference **surfref, const char *symbol) {
	typedef cudaError_t (* pFuncType)(const struct surfaceReference **surfref,
			const char *symbol);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetSurfaceReference");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(surfref, symbol));
}

// --------------------------
cudaError_t cudaDriverGetVersion(int *driverVersion) {
	typedef cudaError_t (* pFuncType)(int *driverVersion);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaDriverGetVersion");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(driverVersion));

}
cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
	typedef cudaError_t (* pFuncType)(int *runtimeVersion);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaRuntimeGetVersion");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(runtimeVersion));
}

// --------------------------
cudaError_t cudaGetExportTable(const void **ppExportTable,
		const cudaUUID_t *pExportTableId) {
	typedef cudaError_t (* pFuncType)(const void **ppExportTable,
			const cudaUUID_t *pExportTableId);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetExportTable");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(ppExportTable, pExportTableId));
}

// ----------------------------------------
//! Unlisted CUDA calls for state registration
// ----------------------------------------


/**
 * The implementation is taken from __nvback_cudaRegisterFatBinary_rpc from
 * remote_api_wrapper and from cudart.c __cudaRegisterFatBinary
 *
 * @return cuda_err set to cudaErrorMemoryAllocation if I cannot calloc the packet
 *
 */
void** __cudaRegisterFatBinary(void* fatC) {

/*
//	pPacket->thr_id = pthread_self();
//	pPacket->args[0].argp = fatC;
//	pPacket->args[1].argi = getFatRecSize(fatC, &entries_cached);
//	pPacket->flags = CUDA_request | CUDA_Copytype;
//	pPacket->method_id = __CUDA_REGISTER_FAT_BINARY;

	// send the packet
//	__nvback_cudaRegisterFatBinary_rpc(pPacket);

	// now you can
//	free(pPacket);
//	return NULL;


	// -----------------------
	cache_num_entries_t nentries = {0, 0, 0, 0, 0};
	int size;
	// let's look like it will be __cudaFatCudaBinary
	// in fact we are allocating the contiguous area of memory that should
	// be treated as a void*, but we want to make it look like the structure
	// __cudaFatCudaBinary; that's why we put it as a __cudaFatCudaBinary
	// and not *void
	__cudaFatCudaBinary * pDestFatC;
	__cudaFatCudaBinary *pSrcFatC = (__cudaFatCudaBinary *)fatC;
	cuda_packet_t *pPacket;


	if (fatC == NULL) {
		printd(DBG_ERROR, "%s, Null CUDA fat binary. Have to exit\n", __FUNCTION__);
		exit(ERROR);
	}


	// Now make a packet and send
	pPacket = callocCudaPacket(__FUNCTION__, &cuda_err);
	if( pPacket == NULL ){
		exit(ERROR);
	}

	// @todo this should be hidden from this implementation
	// it should go to the copyFatBinary
	size = getFatRecSize(pSrcFatC, &nentries);

	// Make it a multiple of page size
	printd(DBG_INFO, "%s: Size is %d\n", __FUNCTION__, size);

	// @todo it should be problably done in serializeFatBinary function
	// allocate the memory for the
	pDestFatC = (__cudaFatCudaBinary *) malloc(size);
	if( mallocCheck( pDestFatC, __FUNCTION__, "During copy of CUDA FAT BINARY" ) != OK){
		cuda_err = cudaErrorMemoryAllocation;
		exit(ERROR);
	}

	pDestFatC = serializeFatBinary(pSrcFatC,&nentries, pDestFatC);

	// Now make a packet and send
	pPacket = (cuda_packet_t *)calloc(1, sizeof(cuda_packet_t));
	if (pPacket == NULL) {
		cuda_err = cudaErrorMemoryAllocation;
		exit(-1);
	}
	pPacket->thr_id = pthread_self();
	pPacket->args[0].argp = pDestFatC;
	pPacket->args[1].argi = size;

	pPacket->flags = CUDA_request | CUDA_Copytype;
	pPacket->method_id = __CUDA_REGISTER_FAT_BINARY;

	if( __nvback_cudaRegisterFatBinary_rpc(pPacket) != CUDA_SUCCESS ){
		printd(DBG_DEBUG, "%s: Problems with rpc\n", __FUNCTION__);
	}

	free(pPacket);
	// Now we can free the original address since the backend must have
	// copied all the data into new buffer
	free(pDestFatC);

	// @todo this Vishakha told me that should be invoked otherwise the program
	// doesn't work. So let's see. This doesn't harm if you call unregister
	// so we will call it, otherwise we need to return something
	// we get from the packet.
*/
	static void** (*func)(void* fatC) = NULL;
	char *error;

	if (!func) {
		func = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	return (func(fatC));

}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
	typedef void** (* pFuncType)(void** fatCubinHandle);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle));
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize) {
	typedef void** (* pFuncType)(void** fatCubinHandle, const char* hostFun,
			char* deviceFun, const char* deviceName, int thread_limit,
			uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterFunction");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
			bid, bDim, gDim, wSize));
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global) {
	typedef void** (* pFuncType)(void **fatCubinHandle, char *hostVar,
			char *deviceAddress, const char *deviceName, int ext, int vsize,
			int constant, int global);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterVar");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, vsize,
			constant, global));

}

void __cudaRegisterTexture(void** fatCubinHandle,
		const struct textureReference* hostVar, const void** deviceAddress,
		const char* deviceName, int dim, int norm, int ext) {
	typedef void** (* pFuncType)(void** fatCubinHandle,
			const struct textureReference* hostVar, const void** deviceAddress,
			const char* deviceName, int dim, int norm, int ext);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterTexture");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext));
}

void __cudaRegisterShared(void** fatCubinHandle, void** devicePtr) {
	typedef void** (* pFuncType)(void** fatCubinHandle, void** devicePtr);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterShared");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle, devicePtr));
}
// ---------------------------
// end
// ---------------------------






/*
 void *malloc(size_t size)
 {
 static void * (*func)(size_t size) = NULL;
 char *error;
 void *addr;

 printf("******malloc\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "malloc");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return NULL;
 }
 }

 addr = func(size);

 printf("%s: size %ld, acquired addr: %p\n",
 __FUNCTION__, size, addr);
 return addr;
 }

 void free(void *ptr)
 {
 static void (*func)(void *ptr) = NULL;
 char *error;

 printf("******\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "free");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 }
 }

 printf("%s: release addr: %p\n",
 __FUNCTION__, ptr);
 func(ptr);
 }
 */
/*cudaError_t cudaMalloc(void **devPtr, size_t size)
 {
 static cudaError_t (*func) (void **ptrs, size_t sz) = NULL; // for the real call
 char *error;

 printf("******cudaMalloc\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaMalloc");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }

 printf("%s: devPtr %p, size %ld\n", __FUNCTION__, devPtr, size);
 return(func(devPtr, size));
 }

 cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
 cudaStream_t stream)
 {
 static cudaError_t (*func) (dim3 gridDim, dim3 blockDim, size_t sharedMem, \
				cudaStream_t stream) = NULL; // for the real call
 char *error;

 printf("******cudaConfigureCall\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaConfigureCall");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }

 printf("gridDim.x = %d, .y = %d, .z = %d\n",
 gridDim.x, gridDim.y, gridDim.z);
 printf("blockDim.x = %d, .y = %d, .z = %d\n",
 blockDim.x, blockDim.y, blockDim.z);
 printf("sharedMem %ld, stream %d\n", sharedMem, stream);
 return (func(gridDim, blockDim, sharedMem, stream));
 }


 cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
 {
 static cudaError_t (*func) (const void *ag, size_t sz, size_t off) = NULL;
 char *error;

 printf("******cudaSetupArgument\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaSetupArgument");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }
 printf("arg %p, size %ld, offset %ld\n", arg, size, offset);
 printf("*arg 0x%lx\n", *((unsigned long *)arg));
 //   exit(0);
 return (func(arg, size, offset));
 }


 cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
 enum cudaMemcpyKind kind)
 {
 static cudaError_t (*func)(void *dst, const void *src, size_t count,
 enum cudaMemcpyKind kind)= NULL;
 char *error;

 printf("******cudaMemcpy\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaMemcpy");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }

 printf("dst:%p, src %p, count %ld, kind %d\n",
 dst, src, count, kind);
 return (func(dst, src, count, kind));
 }

 cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
 {
 static cudaError_t (*func)(const char *sym, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind) = NULL;
 char *error;

 printf("******cudaMemcpyToSymbol\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaMemcpyToSymbol");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }

 printf("%s: symbol %p:%s, src %p:%s, count %ld, offset %ld, kind %d\n",
 __FUNCTION__, symbol, symbol, src, (char *)src, count, offset, kind);
 return (func(symbol, src, count, offset, kind));
 }


 cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind )
 {
 printf("******cudaMemcpyFromSymbol\n");
 return 0;
 }

 #include "nvfront_cuda_fat_format.h"

 // Cache some useful information here
 typedef struct {
 int nptxs;
 int ncubs;
 int ndebs;
 int nrecs;
 } cache_num_entries_t;


 int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num)
 {
 int size = 0, i;
 __cudaFatPtxEntry *tempPtx;
 __cudaFatCubinEntry *tempCub;
 __cudaFatDebugEntry *tempDeb;
 __cudaFatSymbol *tempExp, *tempImp;
 __cudaFatCudaBinary *tempRec;

 // Adding all fields independently to make sure we get platform
 // dependent size of everything and remain agnostic to minor
 // data type changes for these fields
 size += sizeof(fatCubin->magic);
 size += sizeof(fatCubin->version);
 size += sizeof(fatCubin->gpuInfoVersion);
 // The char * are supposed to be null terminated
 size += strlen(fatCubin->key) + 1;  // always 1 extra for the null char
 size += strlen(fatCubin->ident) + 1;
 size += strlen(fatCubin->usageMode) + 1;

 size += sizeof( __cudaFatPtxEntry *);  // space to store addr
 tempPtx = fatCubin->ptx;
 i = 0;
 if (tempPtx != NULL) {
 while (!(tempPtx[i].gpuProfileName == NULL && tempPtx[i].ptx == NULL)) {
 size += sizeof( __cudaFatPtxEntry);  // space to store elems at the addr
 size += strlen(tempPtx[i].gpuProfileName) + 1;  // size of elements
 size += strlen(tempPtx[i].ptx) + 1;
 i++;
 num->nptxs++;
 }
 // Account for the null entries but no strlen required
 size += sizeof( __cudaFatPtxEntry);  // space to store elems at the addr
 num->nptxs++;	// for the null entry
 }
 size += sizeof( __cudaFatCubinEntry *);  // space to store addr
 tempCub = fatCubin->cubin;
 i = 0;
 if (tempCub != NULL) {
 while (	!(tempCub[i].gpuProfileName == NULL && tempCub[i].cubin == NULL)) {
 size += sizeof( __cudaFatCubinEntry);  // space to store elems at the addr
 size += strlen(tempCub[i].gpuProfileName) + 1;  // size of elements
 size += strlen(tempCub[i].cubin) + 1;
 num->ncubs++;
 i++;
 }
 size += sizeof( __cudaFatCubinEntry);  // space to store elems at the addr
 num->ncubs++;
 }
 size += sizeof( __cudaFatDebugEntry *);  // space to store addr
 tempDeb = fatCubin->debug;
 i = 0;
 if (tempDeb != NULL) {
 while (!(tempDeb[i].gpuProfileName == NULL && tempDeb[i].debug == NULL)) {
 size += sizeof( __cudaFatDebugEntry);  // space to store elems at the addr
 size += strlen(tempDeb[i].gpuProfileName) + 1;  // size of elements
 size += strlen(tempDeb[i].debug) + 1;
 num->ndebs++;
 i++;
 }
 size += sizeof( __cudaFatDebugEntry);  // space to store elems at the addr
 num->ndebs++;
 }

 size += sizeof(fatCubin->debugInfo);
 size += sizeof(fatCubin->flags);

 tempExp = fatCubin->exported;
 #ifndef CORRECTWAY
 size += sizeof(__cudaFatSymbol *);  // space to store addr
 #else  // there can be some issue with the ptr addr which can cause the code to crash
 // Therefore hacking
 while (tempExp != NULL && tempExp->name != NULL) {
 size += sizeof(__cudaFatSymbol *);  // space to store addr
 size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
 size += strlen(tempExp->name) + 1;  // size of elements
 tempExp++;
 }
 #endif
 tempImp = fatCubin->imported;
 #ifndef CORRECTWAY
 size += sizeof(__cudaFatSymbol *);  // space to store addr
 #else  // there can be some issue with the ptr addr which can cause the code to crash
 // Therefore hacking
 while (tempImp != NULL && tempImp->name != NULL) {
 size += sizeof(__cudaFatSymbol *);  // space to store addr
 size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
 size += strlen(tempImp->name) + 1;  // size of elements
 tempImp++;
 }
 #endif
 size += sizeof(__cudaFatCudaBinary *);  // space to store addr
 tempRec = fatCubin->dependends;
 i = 0;
 if (tempRec != NULL) {
 while (tempRec[i].ident != NULL) {
 cache_num_entries_t nent = {0};
 size += sizeof(__cudaFatCudaBinary);
 size += get_fat_rec_size(&tempRec[i], &nent);  // space to store elems at the addr
 num->nrecs++;
 i++;
 }
 size += sizeof(__cudaFatCudaBinary);
 num->nrecs++;
 }

 size += sizeof(fatCubin->characteristic);

 //	printf("%s: ident=%s, size found=%d\n", __FUNCTION__, fatCubin->ident, size);
 return size;
 }

 #define GET_LOCAL_POINTER(curr_marker, size, new_marker, dtype) { \
	curr_marker = (char *)((unsigned long)curr_marker + size); \
	curr_marker[size - 1] = 0;  \
	new_marker = dtype(curr_marker); \
}

 void* copyFatBinary(void* fatC, cache_num_entries_t *nentries, void *addr)
 {
 __cudaFatCudaBinary *fatCubin = (__cudaFatCudaBinary *)fatC;
 __cudaFatCudaBinary *nFatCubin;
 char *curr;
 __cudaFatCudaBinary *tempRec, *nTempRec;
 __cudaFatPtxEntry *tempPtx, *nTempPtx;
 __cudaFatCubinEntry *tempCub, *nTempCub;
 __cudaFatDebugEntry *tempDeb, *nTempDeb;
 int len, i;

 // Now make a copy in a contiguous buffer
 // Doing it step by step because we need to allocate pointers
 // as we go and there isnt much that will change by using memcpy
 nFatCubin = (__cudaFatCudaBinary *)addr;
 printf("%s: nFatCubin addr = %p\n", __FUNCTION__, addr);
 nFatCubin->magic = fatCubin->magic;
 nFatCubin->version = fatCubin->version;
 nFatCubin->gpuInfoVersion = fatCubin->gpuInfoVersion ;

 curr = (char *)((unsigned long)nFatCubin);

 // \todo Some repeat work. Can cache these lengths
 GET_LOCAL_POINTER(curr, sizeof(__cudaFatCudaBinary), nFatCubin->key, (char *));
 strcpy(nFatCubin->key, fatCubin->key);
 GET_LOCAL_POINTER(curr, (strlen(fatCubin->key) + 1), nFatCubin->ident, (char *));
 strcpy(nFatCubin->ident, fatCubin->ident);
 GET_LOCAL_POINTER(curr, (strlen(fatCubin->ident) + 1), nFatCubin->usageMode, (char *));
 strcpy(nFatCubin->usageMode, fatCubin->usageMode);

 // Ptx block
 GET_LOCAL_POINTER(curr, (strlen(fatCubin->usageMode) + 1), nFatCubin->ptx, (__cudaFatPtxEntry *));
 len = nentries->nptxs * sizeof(__cudaFatPtxEntry);
 tempPtx = fatCubin->ptx;
 nTempPtx = nFatCubin->ptx;
 if (tempPtx != NULL) {
 for (i = 0; i < nentries->nptxs; ++i) {
 if (tempPtx[i].gpuProfileName != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempPtx[i].gpuProfileName, (char *));
 strcpy(nTempPtx[i].gpuProfileName, tempPtx[i].gpuProfileName);
 len = strlen(tempPtx[i].gpuProfileName) + 1;
 }
 else
 nTempPtx[i].gpuProfileName = 0;
 if (tempPtx[i].ptx != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempPtx[i].ptx, (char *));
 strcpy(nTempPtx[i].ptx, tempPtx[i].ptx);
 len = strlen(tempPtx[i].ptx) + 1;
 }
 else
 nTempPtx[i].ptx = 0;
 }
 }
 else
 nFatCubin->ptx = NULL;

 // Cubin block
 GET_LOCAL_POINTER(curr, len, nFatCubin->cubin, (__cudaFatCubinEntry *));
 len = nentries->ncubs * sizeof(__cudaFatCubinEntry);
 tempCub = fatCubin->cubin;
 nTempCub = nFatCubin->cubin;
 if (tempCub != NULL) {
 for (i = 0; i < nentries->ncubs; ++i) {
 if (tempCub[i].gpuProfileName != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempCub[i].gpuProfileName, (char *));
 strcpy(nTempCub[i].gpuProfileName, tempCub[i].gpuProfileName);
 len = strlen(tempCub[i].gpuProfileName) + 1;
 }
 else
 nTempCub[i].gpuProfileName = 0;
 if (tempCub[i].cubin != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempCub[i].cubin, (char *));
 strcpy(nTempCub[i].cubin, tempCub[i].cubin);
 len = strlen(tempCub[i].cubin) + 1;
 }
 else
 nTempCub[i].cubin = 0;
 }
 }
 else
 nFatCubin->cubin = NULL;

 // Debug block
 GET_LOCAL_POINTER(curr, len, nFatCubin->debug, (__cudaFatDebugEntry *));
 len = nentries->ndebs * sizeof(__cudaFatDebugEntry);
 tempDeb = fatCubin->debug;
 nTempDeb = nFatCubin->debug;
 if (tempDeb != NULL) {
 for (i = 0; i < nentries->ndebs; ++i) {
 if (tempDeb[i].gpuProfileName != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempDeb[i].gpuProfileName, (char *));
 strcpy(nTempDeb[i].gpuProfileName, tempDeb[i].gpuProfileName);
 len = strlen(tempDeb[i].gpuProfileName) + 1;
 }
 else
 nTempDeb[i].gpuProfileName = 0;
 if (tempDeb[i].debug != NULL) {
 GET_LOCAL_POINTER(curr, len, nTempDeb[i].debug, (char *));
 strcpy(nTempDeb[i].debug, tempDeb[i].debug);
 len = strlen(tempDeb[i].debug) + 1;
 }
 else
 nTempDeb[i].debug = 0;
 }
 }
 else
 nFatCubin->debug = NULL;

 nFatCubin->debugInfo = fatCubin->debugInfo;
 nFatCubin->flags = fatCubin->flags;
 #ifndef CORRECTWAY
 nFatCubin->exported = fatCubin->exported;
 nFatCubin->imported = fatCubin->imported;
 #else
 GET_LOCAL_POINTER(curr, (strlen(fatCubin->debug->debug) + 1), nFatCubin->exported, (__cudaFatSymbol *));
 GET_LOCAL_POINTER(curr, sizeof(__cudaFatSymbol), nFatCubin->exported->name, (char *));
 strcpy(nFatCubin->exported->name, fatCubin->exported->name);

 GET_LOCAL_POINTER(curr, (strlen(fatCubin->exported->name) + 1), nFatCubin->imported, (__cudaFatSymbol *));
 GET_LOCAL_POINTER(curr, sizeof(__cudaFatSymbol), nFatCubin->imported->name, (char *));
 strcpy(nFatCubin->imported->name, fatCubin->imported->name);
 #endif
 GET_LOCAL_POINTER(curr, len, nFatCubin->dependends, (__cudaFatCudaBinary *));
 len = nentries->nrecs * sizeof(__cudaFatCudaBinary);
 tempRec = fatCubin->dependends;
 nTempRec = nFatCubin->dependends;
 cache_num_entries_t nent = {0};
 if (tempRec != NULL) {
 // \todo This part definitely needs testing.
 for (i = 0; i < nentries->nrecs; ++i) {
 // \todo Right now, this is completely wrong. Every new
 // element will end up overwriting the previous one bec
 // copyFatBinary in this case does  not know where to
 // start new allocations from.
 GET_LOCAL_POINTER(curr, len, curr, (char *));
 int size = get_fat_rec_size(&tempRec[i], &nent);
 copyFatBinary(tempRec, &nent, &nTempRec[i]);
 len = size;
 }
 }
 else
 nFatCubin->dependends = NULL;  // common case

 nFatCubin->characteristic = fatCubin->characteristic;

 return nFatCubin;
 }

 #define MY_PAGE_SIZE 4096  // temporary

 void** __cudaRegisterFatBinary(void* fatC)
 {
 static void** (*func)(void* fatC) = NULL;
 __cudaFatCudaBinary *fatCubin = (__cudaFatCudaBinary *)fatC;
 cache_num_entries_t nentries = {0};
 char *error;
 printf("******__cudaRegisterFatBinary\n");
 int size = get_fat_rec_size(fatCubin, &nentries);
 // Make it a multiple of page size
 size = (size + MY_PAGE_SIZE - 1) & ~(MY_PAGE_SIZE - 1);
 printf("%s; Size is %d\n", __FUNCTION__, size);
 __cudaFatCudaBinary *nFatCubin = copyFatBinary(fatCubin, &nentries, malloc(size));
 printf("\tgpu profile old = %s, new %s\n",
 fatCubin->cubin->gpuProfileName,
 nFatCubin->cubin->gpuProfileName);

 if (!func) {
 func = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return NULL;
 }
 }

 // Now push this data to the packets and register this in the backend
 //        return (func(fatC));
 return (func(nFatCubin));
 }

 void __cudaUnregisterFatBinary( void** fatCubinHandle )
 {
 static void (*func)( void** fatCubinHandle) = NULL;
 char *error;

 printf("******__cudaUnRegisterFatBinary\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 }
 }
 func(fatCubinHandle);
 }

 typedef struct {
 void** fatCubinHandle;
 const char* hostFun;
 char* deviceFun;
 const char* deviceName;
 int thread_limit;
 uint3* tid;
 uint3* bid;
 dim3* bDim;
 dim3* gDim;
 int* wSize;
 } reg_func_args_t;

 reg_func_args_t *c_args;

 cudaError_t cudaLaunch(const char *symbol)
 {
 static cudaError_t (*func)(const char *sym) = NULL;
 char *error;

 printf("******cudaLaunch\n");
 if (!func) {
 func = dlsym(RTLD_NEXT, "cudaLaunch");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 return -1;
 }
 }

 printf("%s: symbol %p,%s; launch %s\n", __FUNCTION__, symbol, symbol, c_args->hostFun);
 return (func(c_args->hostFun));
 //return (func(symbol));
 }

 reg_func_args_t *copyRegFuncArgs( void** fatCubinHandle, const char* hostFun,
 char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
 uint3* bid, dim3* bDim, dim3* gDim, int* wSize,
 int *tsize)   // last argument to return the total size
 {
 int size, len;
 char *curr;
 reg_func_args_t *args;

 size = 9 * sizeof(void *);   // Sizeof all the pointer arguments
 size += sizeof(int); // for thread_limit

 // Now for the serializing part
 // fatCubinHandle should be passed as one double pointer. The correct
 // pointing should be handled in Dom0 by the backend code since thats
 // where the binary must have gotten registered
 if (hostFun != NULL)
 size += strlen(hostFun) + 1;
 if (deviceFun != NULL)
 size += strlen(deviceFun) + 1;
 if (deviceName != NULL)
 size += strlen(deviceName) + 1;
 if (tid != NULL)
 size += sizeof(uint3);
 if (bid != NULL)
 size += sizeof(uint3);
 if (bDim != NULL)
 size += sizeof(dim3);
 if (gDim != NULL)
 size += sizeof(dim3);
 if (wSize != NULL)
 size += sizeof(int);

 // Now start the copying
 size = (size + MY_PAGE_SIZE - 1) & ~(MY_PAGE_SIZE - 1);
 args = (reg_func_args_t *)malloc(size);
 args->fatCubinHandle = fatCubinHandle;  // 1

 curr = (char *)((unsigned long)args);
 len = sizeof(reg_func_args_t);
 // \todo Some repeat work. Can cache these lengths
 if (hostFun != NULL) {  // 2
 GET_LOCAL_POINTER(curr, len, args->hostFun, (char *));
 strcpy (args->hostFun, hostFun);
 len = strlen(hostFun) + 1;
 }
 else
 args->hostFun = 0;
 if (deviceFun != NULL) {  // 3
 GET_LOCAL_POINTER(curr, len, args->deviceFun, (char *));
 strcpy (args->deviceFun, deviceFun);
 len = strlen(deviceFun) + 1;
 }
 else
 args->deviceFun = 0;
 if (deviceName != NULL) {  // 4
 GET_LOCAL_POINTER(curr, len, args->deviceName, (char *));
 strcpy (args->deviceName, deviceName);
 len = strlen(deviceName) + 1;
 }
 else
 args->deviceName = 0;

 args->thread_limit = thread_limit;  // 5

 if (tid != NULL) {  // 6
 GET_LOCAL_POINTER(curr, len, args->tid, (uint3 *));
 len = sizeof(uint3);
 memcpy (args->tid, tid, len);
 }
 else
 args->tid = 0;
 if (bid != NULL) {  // 7
 GET_LOCAL_POINTER(curr, len, args->bid, (uint3 *));
 len = sizeof(uint3);
 memcpy (args->bid, bid, len);
 }
 else
 args->bid = 0;

 if (bDim != NULL) {   // 8
 GET_LOCAL_POINTER(curr, len, args->bDim, (dim3 *));
 len = sizeof(dim3);
 memcpy (args->bDim, bDim, len);
 }
 else
 args->bDim = 0;
 if (gDim != NULL) {  // 9
 GET_LOCAL_POINTER(curr, len, args->gDim, (dim3 *));
 len = sizeof(dim3);
 memcpy (args->gDim, gDim, len);
 }
 else
 args->gDim = 0;

 if (wSize != NULL) {  // 10
 GET_LOCAL_POINTER(curr, len, args->wSize, (int *));
 memcpy (args->wSize, wSize, len);
 }
 else
 args->wSize = 0;

 *tsize = size;
 return args;
 }

 void __cudaRegisterFunction( void** fatCubinHandle, const char* hostFun,
 char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
 uint3* bid, dim3* bDim, dim3* gDim, int* wSize )
 {
 static void (*func)( void** fatCH, const char* hFun,
 char* dFun, const char* dName, int tLimit, uint3* tid,
 uint3* bid, dim3* bDim, dim3* gDim, int* wSize ) = NULL;
 char *error;
 int size;  // of all the arguments total

 c_args = copyRegFuncArgs(fatCubinHandle, hostFun, \
			deviceFun, deviceName, \
			thread_limit, tid, bid, bDim, gDim, wSize, &size);

 printf("******__cudaRegisterFunction handle addr:handle=%p:%p, argsize=%d\n",
 fatCubinHandle, *fatCubinHandle, size);
 if (!func) {
 func = dlsym(RTLD_NEXT, "__cudaRegisterFunction");
 if ((error = dlerror()) != NULL) {
 printf("%s\n", error);
 }
 }
 printf("%s: hostFun=%p:%s, deviceFun=%p:%s, devName=%p:%s\n",
 __FUNCTION__, hostFun, hostFun, deviceFun, deviceFun,
 deviceName, deviceName);
 printf("%s_Args: hostFun=%p:%s, deviceFun=%p:%s, devName=%p:%s\n",
 __FUNCTION__, c_args->hostFun, c_args->hostFun, \
		c_args->deviceFun, c_args->deviceFun,
 c_args->deviceName, c_args->deviceName);
 func(c_args->fatCubinHandle, c_args->hostFun, c_args->deviceFun, c_args->deviceName, \
			c_args->thread_limit, c_args->tid, c_args->bid, \
			c_args->bDim, c_args->gDim, c_args->wSize);
 }

 */

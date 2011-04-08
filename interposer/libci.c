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
#include <assert.h>


//! to indicate the error with the dynamic loaded library
static cudaError_t cudaErrorDL = cudaErrorUnknown;

//! Maintain the last error for cudaGetLastError()
static cudaError_t cuda_err = 0;

//! Right now the host where we are connecting to
const char * SERVER_HOSTNAME = "cuda2.cc.gt.atl.ga.us";
//const char * SERVER_HOSTNAME = "prost.cc.gt.atl.ga.us";

#define MAX_REGISTERED_VARS 10
// \todo Clean it up later. Just need to make sure MemcpySymbol does jazz
// only when a variable has been registered
char *reg_host_vars[MAX_REGISTERED_VARS];
static int num_registered_vars = 0;

//! stores information about the fatcubin_info on the client side
static fatcubin_info_t fatcubin_info_rpc;


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
	printf(">>>>>>>>>> Implemented >>>>>>>>>>: %s\n", pSignature);
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
	(*pPacket)->ret_ex_val.err = cudaErrorUnknown;

	return OK;
}

/**
 * sets the method_id, thr_id, flags in the packet structure to default values
 * @param pPacket The packet to be changed
 * @param methodId The method id you want to set
 * @param pSignature The string describing the function signature
 * @return OK everything is fine
 *         ERROR there is a problem with memory and cuda error contains
 *         the error; if calloc gave NULL
 */

int l_remoteInitMetThrReq(cuda_packet_t ** const pPacket,
		const uint16_t methodId, const char* pSignature){
	printf(">>>>>>>>>> Implemented >>>>>>>>>>: %s (id = %d)\n", pSignature, methodId);

	// Now make a packet and send
	if ((*pPacket = callocCudaPacket(pSignature, &cuda_err)) == NULL) {
		return ERROR;
	}

	(*pPacket)->method_id = methodId;
	(*pPacket)->thr_id = pthread_self();
	(*pPacket)->flags = CUDA_request;
	(*pPacket)->ret_ex_val.err = cudaErrorUnknown;

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
	printd(DBG_DEBUG, ">>>>>>>>>> Implemented >>>>>>>>>>: %s (no id)\n", __FUNCTION__);

	return cuda_err;

/*	typedef cudaError_t (* pFuncType)(void);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetLastError");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc()); */
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

	if( l_remoteInitMetThrReq(&pPacket, CUDA_GET_DEVICE_COUNT, __FUNCTION__) == ERROR){
		return cuda_err;
	}

	// send the packet
	if(nvbackCudaGetDeviceCount_rpc(pPacket) == OK ){
		printd(DBG_INFO, "%s: __OK__ the number of devices is %ld. Got from the RPC call\n", __FUNCTION__,
						pPacket->args[0].argi);
		// remember the count number what we get from the remote device
		*count = pPacket->args[0].argi;
		cuda_err = pPacket->ret_ex_val.err;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;

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

	if( l_remoteInitMetThrReq(&pPacket, CUDA_GET_DEVICE_PROPERTIES, __FUNCTION__) == ERROR){
			return cuda_err;
	}

	// override the flags; just following the cudart.c
	// guessing the CUDA_Copytype means that something needs to be copied
	// over the network
	pPacket->flags |= CUDA_Copytype;   // it now should be CUDA_request | CUDA_Copytype

	// @todo (comment) now we are storing this into argp which is of type (void*)
	// please not that in _rpc counterpart we will interpret this as argui
	// which is of type uint64_t (unsigned long), actually I do not understand;
	// I am guessing that maybe because of mixing 32bit and 64bit machines in
	// original remote_gpu and we want to be sure that
	// I am sticking to the original implementation
	pPacket->args[0].argp = (void *) prop; // I do not understand why we do this
	pPacket->args[1].argi = device;   // I understand this
	pPacket->args[2].argi = sizeof(struct cudaDeviceProp); // for driver; I do not understand why we do this

	// send the packet
	if (nvbackCudaGetDeviceProperties_rpc(pPacket) == OK) {
		l_printCudaDeviceProp(prop);
		cuda_err = pPacket->ret_ex_val.err;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
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
	cuda_packet_t *pPacket;

	if( l_remoteInitMetThrReq(&pPacket, CUDA_SET_DEVICE, __FUNCTION__) == ERROR){
		return cuda_err;
	}
	pPacket->args[0].argi = device;

	// send the packet
	if (nvbackCudaSetDevice_rpc(pPacket) == OK) {
		cuda_err = pPacket->ret_ex_val.err;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;


/*	typedef cudaError_t (* pFuncType)(int device);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device)); */
}
cudaError_t cudaGetDevice(int *device) {
	cuda_packet_t * pPacket;

	if (l_remoteInitMetThrReq(&pPacket, CUDA_GET_DEVICE, __FUNCTION__) == ERROR) {
		return cuda_err;
	}

	// send the packet
	if (nvbackCudaGetDevice_rpc(pPacket) == OK) {
		printd(DBG_INFO, "%s: __OK__ RPC call returned: assigned device id = %ld.\n", __FUNCTION__,
				pPacket->args[0].argi);
		// remember the count number what we get from the remote device
		*device = pPacket->args[0].argi;
		cuda_err = pPacket->ret_ex_val.err;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;

/*	typedef cudaError_t (* pFuncType)(int *device);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaGetDevice");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(device)); */
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

	cuda_packet_t *pPacket;

	if( l_remoteInitMetThrReq(&pPacket, CUDA_CONFIGURE_CALL, __FUNCTION__) == ERROR)
		return cuda_err;

	pPacket->args[0].arg_dim = gridDim;
	pPacket->args[1].arg_dim = blockDim;
	pPacket->args[2].argi = sharedMem;
	pPacket->args[3].arg_str = stream;

	printf("gridDim(x,y,z)=%u, %u, %u; blockDim(x,y,z)=%u, %u, %u; sharedMem (size) = %ld; stream =%ld\n",
			pPacket->args[0].arg_dim.x, pPacket->args[0].arg_dim.y, pPacket->args[0].arg_dim.z,
			pPacket->args[1].arg_dim.x, pPacket->args[1].arg_dim.y, pPacket->args[1].arg_dim.z,
			pPacket->args[2].argi, (long unsigned) pPacket->args[3].arg_str);


	// send the packet
	if (nvbackCudaConfigureCall_rpc(pPacket) == OK) {
		// asynchronous call
		cuda_err = cudaSuccess;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__: Return from rpc with the wrong return value.\n", __FUNCTION__);
		// indicate error situation
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;

/*	typedef cudaError_t (* pFuncType)(dim3 gridDim, dim3 blockDim,
			size_t sharedMem, cudaStream_t stream);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaConfigureCall");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(gridDim, blockDim, sharedMem, stream)); */
}
cudaError_t cudaSetupArgument(const void *arg, size_t size,
		size_t offset) {
	cuda_packet_t *pPacket;

	if( l_remoteInitMetThrReq(&pPacket, CUDA_SETUP_ARGUMENT, __FUNCTION__) == ERROR){
				return cuda_err;
	}

	// override the flags; just following the cudart.c
	// guessing the CUDA_Copytype means that something needs to be copied
	// over the network
	pPacket->flags |= CUDA_Copytype; // it now should be CUDA_request | CUDA_Copytype

	// @todo (comment) now we are storing this into argp which is of type (void*)
	// please not that in _rpc counterpart we will interpret this as argui
	// which is of type uint64_t (unsigned long), actually I do not understand;
	// I am guessing that maybe because of mixing 32bit and 64bit machines in
	// original remote_gpu and we want to be sure that
	// I am sticking to the original implementation

	pPacket->args[0].argp = (void *)arg;  // argument to push for a kernel launch
	pPacket->args[1].argi = size;
	pPacket->args[2].argi = offset; // for driver; Offset in argument stack to push new arg

	// send the packet
	if (nvbackCudaSetupArgument_rpc(pPacket) == OK) {
		cuda_err = cudaSuccess;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;


/*	typedef cudaError_t (* pFuncType)(const void *arg, size_t size,
			size_t offset);
	static pFuncType pFunc = NULL;

	l_printFuncSig(__FUNCTION__);

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaSetupArgument");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	return (pFunc(arg, size, offset)); */
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
	cuda_packet_t *pPacket;

	if( l_remoteInitMetThrReq(&pPacket, CUDA_LAUNCH, __FUNCTION__) == ERROR){
				return cuda_err;
	}
	pPacket->args[0].argcp = (char *)entry;

	printf("%s, entry: %s\n", __FUNCTION__, entry);
	// send the packet
	if (nvbackCudaLaunch_rpc(pPacket) == OK) {
		cuda_err = cudaSuccess;
	} else {
		printd(DBG_ERROR, "%s.%d: __ERROR__: Return from rpc with the wrong return value.\n", __FUNCTION__, __LINE__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	}

	free(pPacket);

	return cuda_err;


/*	typedef cudaError_t (* pFuncType)(const char *entry);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaLaunch");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	l_printFuncSig(__FUNCTION__);

	return (pFunc(entry)); */
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

	if( l_remoteInitMetThrReq(&pPacket, CUDA_MALLOC, __FUNCTION__) == ERROR){
		return cuda_err;
	}

	pPacket->args[0].argdp = devPtr;
	pPacket->args[1].argi = size;

	printf("\ndevPtr %p, *devPtr %p, size %ld\n", devPtr, *devPtr, size);

	if(nvbackCudaMalloc_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s: __ERROR__: Return from the RPC\n", __FUNCTION__);
		cuda_err = cudaErrorMemoryAllocation;
		*devPtr = NULL;
	} else {
		printd(DBG_INFO, "%s: __OK__:  Return from the RPC call DevPtr %p\n", __FUNCTION__,
				pPacket->args[0].argp);
		// unpack what we have got from the packet
		*devPtr = pPacket->args[0].argp;
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);
	return cuda_err;

	/*
	typedef cudaError_t (* pFuncType)(void **devPtr, size_t size);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMalloc");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}

	return (pFunc(devPtr, size)); */
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

	if( l_remoteInitMetThrReq(&pPacket, CUDA_FREE, __FUNCTION__) == ERROR){
				return cuda_err;
	}
	pPacket->args[0].argp = devPtr;

	// send the packet
	if(nvbackCudaFree_rpc(pPacket) == OK ){
		printd(DBG_DEBUG, "%s: __OK__ The used pointer %p\n", __FUNCTION__,
						pPacket->args[0].argp);
		cuda_err = cudaSuccess;
	} else {
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
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
	cuda_packet_t *pPacket;

	// you need to setup a method id individually
	if( l_remoteInitMetThrReq(&pPacket, ERROR, __FUNCTION__) == ERROR){
		return cuda_err;
	}

	pPacket->args[0].argp = dst;
	pPacket->args[1].argp = (void *)src;
	pPacket->args[2].argi = count;
	pPacket->args[3].argi = kind;
	pPacket->flags |= CUDA_Copytype;

	// send the packet
	if(nvbackCudaMemcpy_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		printd(DBG_DEBUG, "%s: __OK__ Return from RPC.\n", __FUNCTION__);
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);

	return cuda_err;

	/* typedef cudaError_t (* pFuncType)(void *dst, const void *src, size_t count,
			enum cudaMemcpyKind kind);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "cudaMemcpy");

		if (l_handleDlError() != 0)
			return cudaErrorDL;
	}
	l_printFuncSig(__FUNCTION__);

	return (pFunc(dst, src, count, kind)); */
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

void** pFatBinaryHandle = NULL;


/**
 * The implementation is taken from __nvback_cudaRegisterFatBinary_rpc from
 * remote_api_wrapper and from cudart.c __cudaRegisterFatBinary
 *
 * @return cuda_err set to cudaErrorMemoryAllocation if I cannot calloc the packet
 *
 */
void** __cudaRegisterFatBinary(void* fatC) {
	cuda_packet_t * pPacket;
	// here we will store the number of entries to spare counting again and again
	// @todo might be unimportant
	cache_num_entries_t entries_cached = {0, 0, 0, 0, 0, 0, 0};
	// the size of the packet for cubin
	int fb_size;

	// In original version of the code we created a kind of a structure
	// in a contiguous thing
	// in fact we are allocating the contiguous area of memory that should
	// be treated as a void*, but we want to make it look like the structure
	// __cudaFatCudaBinary; that's why we put it as a __cudaFatCudaBinary
	// and not *void; it will be the serialized version of fatC

	// the original cubin to get rid of casting to __cudaFatCudaBinary
	__cudaFatCudaBinary * pSrcFatC = (__cudaFatCudaBinary *)fatC;
	// the place where the packed fat binary will be stored
	char * pPackedFat = NULL;

	if (fatC == NULL) {
		printd(DBG_ERROR, "%s: __ERROR__, Null CUDA fat binary. Have to exit\n", __FUNCTION__);
		exit(ERROR);
	}

	// allocate and initialize a packet
	if( l_remoteInitMetThrReq(&pPacket, __CUDA_REGISTER_FAT_BINARY, __FUNCTION__) == ERROR){
		exit(ERROR);
	}

	fb_size = getFatRecPktSize(pSrcFatC, &entries_cached);
	printd(DBG_DEBUG, "%s, FatCubin size: %d\n", __FUNCTION__,fb_size);
	l_printFatBinary(pSrcFatC);

	pPackedFat = (char*) malloc(fb_size);

	if( mallocCheck(pPackedFat, __FUNCTION__, NULL) == ERROR ){
		exit(ERROR);
	}

	if( packFatBinary(pPackedFat, pSrcFatC, &entries_cached) == ERROR ){
		exit(ERROR);
	}

	// now update the packets information
	pPacket->flags |= CUDA_Copytype;
	pPacket->args[0].argp = pPackedFat;			// start of the request buffer
	pPacket->args[1].argi = fb_size;			// the size of the request buffer

	printd(DBG_DEBUG, "pPackedFat, pPacket->args[0].argp = %p, %ld\n",
			pPacket->args[0].argp, pPacket->args[1].argi);
	// send the packet
	if (__nvback_cudaRegisterFatBinary_rpc(pPacket) != OK) {
		printd(DBG_ERROR, "__ERROR__: Return from rpc with the wrong return value.\n");
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		// get the response, get the fat cubin handle
		fatcubin_info_rpc.fatCubin = pSrcFatC;
		fatcubin_info_rpc.fatCubinHandle = pPacket->ret_ex_val.handle;

		// @todo maybe I am wrong with this assert, maybe it should
		// be what is got from rpc call: args[1].argp which is
		// the fatCubin of the remote server
		// actually I think it doesn't matter, since to unregister
		// you need a handler
		assert(fatC == fatcubin_info_rpc.fatCubin);
		printd(DBG_INFO, "fatcubin_info_rpc.fatCubinHandle = %p\n", fatcubin_info_rpc.fatCubinHandle);
		printd(DBG_INFO, "fatcubin_info_rpc.fatCubin = %p\n", fatcubin_info_rpc.fatCubin);
	}

	free(pPacket);

	return fatcubin_info_rpc.fatCubinHandle;
/*
// -------------
	static void** (*func)(void* fatC) = NULL;
	//char *error;

	if (!func) {
		func = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	//l_printFuncSig(__FUNCTION__);
	return (func(fatC));
*/
}

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
	cuda_packet_t * pPacket;

	if( l_remoteInitMetThrReq(&pPacket, __CUDA_UNREGISTER_FAT_BINARY, __FUNCTION__) == ERROR){
		exit(ERROR);
	}

	// update packet
	pPacket->args[0].argdp = fatcubin_info_rpc.fatCubinHandle;

	if (__nvback_cudaUnregisterFatBinary_rpc(pPacket) != OK) {
		printd(DBG_ERROR, "%s.%d: __ERROR__ Return from rpc with the wrong return value.\n", __FUNCTION__, __LINE__);
		// @todo some cleaning or setting cuda_err
		cuda_err = cudaErrorUnknown;
	} else {
		// @todo do nothing (?)
		printd(DBG_ERROR, "%s.%d: __OK__ Return from rpc with ok value.\n", __FUNCTION__, __LINE__);
	}

	free(pPacket);

	// @todo Uninit the library as well
	//CUV_EXIT();

	// local housekeeping for variables et al
	num_registered_vars = 0;

	// -----------------
/*	typedef void** (* pFuncType)(void** fatCubinHandle);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaUnregisterFatBinary");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle)); */
}


void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize) {
	cuda_packet_t * pPacket;

	if( l_remoteInitMetThrReq(&pPacket, __CUDA_REGISTER_FUNCTION, __FUNCTION__) == ERROR){
		exit(ERROR);
	}

	l_printRegFunArgs(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,tid,
			bid, bDim, gDim, wSize);

	int size = 0;

	char * p = packRegFuncArgs(fatCubinHandle, hostFun,
			deviceFun, deviceName, thread_limit, tid,
			 bid, bDim, gDim, wSize, &size);

	if( !p ){
		printd(DBG_ERROR, "%s: __ERROR__ Problems with allocating the memory. Quitting ... \n", __FUNCTION__);
		exit(ERROR);
	}
	// update packet; point to the buffer from which you will
	// take data to send over the network
	pPacket->flags |= CUDA_Copytype;
	pPacket->args[0].argp = p;			// buffer pointer
	pPacket->args[1].argi = size;       // size of the buffer

	//(void *) packet->args[0].argui, packet->args[1].argi

	if(__nvback_cudaRegisterFunction_rpc(pPacket) != OK ){
		printd(DBG_ERROR, "%s: __ERROR__: Return from the RPC with an error\n", __FUNCTION__);
		cuda_err = cudaErrorUnknown;
	} else {
		// do nothing;
		// @todo don't you need to put some stuff
		// to fatcubin_info_rpc like the functions registered?
		cuda_err = pPacket->ret_ex_val.err;
	}

	free(pPacket);

	// --------------
/*	typedef void** (* pFuncType)(void** fatCubinHandle, const char* hostFun,
			char* deviceFun, const char* deviceName, int thread_limit,
			uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
	static pFuncType pFunc = NULL;

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterFunction");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	//l_printFuncSig(__FUNCTION__);

	(pFunc(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
			bid, bDim, gDim, wSize)); */
	return;
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

/**
 * @file libci.c
 *
 * @date Feb 27, 2011
 * @author Magda Slawinska, magg@gatech.edu
 * @author Alex Merritt, merritt.alex@gatech.edu
 *
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
 * FIXME Make this library thread-safe.
 *
 * As of right now, the format of the shared memory regions with the backend is
 * as follows:
 *
 * 	byte	data
 * 	0		struct cuda_packet
 * 	[n		data associated with packet, void *]
 *
 * Any data associated with a packet is generally a result of a pointer
 * argument. Either data is being copied in, or used as an output argument.
 * Pointer arguments are either associated with a known data type, in which case
 * no extra metadata is needed to describe the size of data, or a copy of
 * generic memory. For the latter, extra arguments are present in the argument
 * list that describe the region size.
 *
 * When constructing packets, arguments that represent pointers are replaced
 * with values representing offsets into the shared memory region. Actual
 * addresses are useless, as they are virtual (processes have different virtual
 * address spaces). See cudaGetDeviceCount as a simple example.
 */

// if this file is .c, if you do not define _GNU_SOURCE then it complains
// that RTLD_NEXT is undeclared
// if this file is  .cpp, then you get warning "_GNU_SOURCE" redefined
#define _GNU_SOURCE

/*-------------------------------------- INCLUDES ----------------------------*/

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
#include "libciutils.h"
#include <assert.h>

#include <glib.h>		// for GHashTable
#include "kidron_common_s.h" // for ini file

#include <util/x86_system.h> // memory barriers
#include <assembly.h>
#include <util/compiler.h>
#include <common/libregistration.h>

#include "config.h"

/*-------------------------------------- EXTERNS -----------------------------*/

// from kidron_common_f.c
extern int ini_getLocal(const ini_t* pIni);
extern int ini_getIni(ini_t* pIni);
extern int ini_freeIni(ini_t* pIni);

/*-------------------------------------- DEFINITIONS -------------------------*/

/**
 * Struct representing the state associated with registration with the backend
 * process. Just a place to store this local collection of information. Each
 * registration ID is associated with a different memory region. TODO Currently
 * the registration library does not support more than one. When it does, each
 * may be used by a different application thread to store packets.
 * TODO Use a thread ID to index into the correct shm.
 */
struct backend_reg {
	gboolean	has_registered;
	regid_t		regs[5];		//! Library-Backened registration IDs (for SHM)
	void		*shm[5];		//! Memory regions shared with the backend
	size_t		sizes[5];		//! Sizes of each region.
};

#define NEED_REGISTRATION	(unlikely(be_reg.has_registered == FALSE))
#define NEED_UNREGISTRATION	(be_reg.has_registered == TRUE)

#define PRINT_FUNC	printf("CAUGHT %s\n", __func__)

/*-------------------------------------- INTENRAL STATE ----------------------*/

/*
 * These variables are 'global' with respect to this file only.
 */

//! to indicate the error with the dynamic loaded library
static cudaError_t cudaErrorDL = cudaErrorUnknown;

//! State machine for cudaGetLastError()
static cudaError_t cuda_err = 0;

static struct backend_reg be_reg;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

//! This appears in some values of arguments. I took this from
//! /opt/cuda/include/cuda_runtime_api.h ! It looks as this comes from a default
//! value (dv)
#if !defined(__dv)
#	if defined(__cplusplus)
#		define __dv(v) \
        		= v
#	else
#		define __dv(v)
#	endif
#endif

static int registerWithBackend(void) {
	int err;

	if (be_reg.has_registered == FALSE) {
		err = reg_lib_init();
		if (err < 0) {
			printd(DBG_ERROR, "Could not initialize registration with backend\n");
			goto fail;
		}
		be_reg.has_registered = TRUE;
		// request new mmap region, initialize with hint struct
		be_reg.regs[0] = reg_lib_connect();
		if (be_reg.regs[0] < 0) {
			printd(DBG_ERROR, "Could not connect\n");
			goto fail;
		}
		be_reg.shm[0] = reg_lib_get_shm(be_reg.regs[0]);
		be_reg.sizes[0] = reg_lib_get_shm_size(be_reg.regs[0]);
		printd(DBG_DEBUG, "id=%d shm=%p sz=%lu\n",
				be_reg.regs[0], be_reg.shm[0], be_reg.sizes[0]);
	}
	printd(DBG_INFO, "complete\n");
	return 0;

fail:
	return -1;
}

static int unregisterWithBackend(void) {
	int err;
	if (be_reg.has_registered == TRUE) {
		err = reg_lib_disconnect(be_reg.regs[0]);
		if (err < 0) {
			printd(DBG_ERROR, "Could not disconnect\n");
			goto fail;
		}
		err = reg_lib_shutdown();
		if (err < 0) {
			printd(DBG_ERROR, "Could not shutdown\n");
			goto fail;
		}
		be_reg.has_registered = FALSE;
	}
	printd(DBG_INFO, "complete\n");
	return 0;

fail:
	return -1;
}

/*-------------------------------------- INTERPOSING API ---------------------*/

/* Updates needed in the future:
 *
 * TODO Each function need to choose which shm region it must use. This will
 * depend on the thread ID making the call.
 *
 * TODO Support placing multiple calls into a shm region. This will require
 * using some sort of queueing data structure within it.
 */

cudaError_t cudaGetDeviceCount(int *count) {
	int err;
	volatile struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];
	PRINT_FUNC;

	// Most CUDA programs call __registerFatBinary first, this is here to catch
	// deviceQuery in the SDK. Better would be to check for registration in
	// every function call.
	if (NEED_REGISTRATION) {
		err = registerWithBackend();
		if (err < 0) {
			fprintf(stderr, "Error registering with backend\n");
			return -1;
		}
		shmpkt = (struct cuda_packet *)be_reg.shm[0];
	}

	/* Directly write the cuda packet into the shared memory region. Indicate
	 * where the output argument's data should be stored.
	 *
	 *   shm	Offset
	 * +------+ 0
	 * + cuda +
	 * + pkt  +
	 * +------+	sizeof(struct cuda_packet)
	 * + int  + 
	 * +------+ sizeof(int) + sizeof(struct cuda_packet)
	 *
	 * Give the packet one argument indicating the offset into the shared memory
	 * region where it expects the value of 'count' to reside. This will be
	 * copied into the user-provided address before returning. For functions
	 * with additional pointer arguments, append them to the bottom and do
	 * additional copying before returning.
	 */

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = CUDA_GET_DEVICE_COUNT;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(struct cuda_packet); // offset
	shmpkt->flags = CUDA_PKT_REQUEST; // set this last FIXME sink spins on this

	wmb(); // flush writes from caches

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();

	*count = *((int *)((void *)shmpkt + shmpkt->args[0].argull));
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
	struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];
	struct cudaDeviceProp *prop_shm = NULL;
	PRINT_FUNC;

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = CUDA_GET_DEVICE_PROPERTIES;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(struct cuda_packet); // offset
	shmpkt->args[1].argll = device;
	shmpkt->flags = CUDA_PKT_REQUEST;

	wmb();

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();

	prop_shm = (struct cudaDeviceProp*)((void*)shmpkt + shmpkt->args[0].argull);
	memcpy(prop, prop_shm, sizeof(struct cudaDeviceProp));
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetDevice(int device) {
	struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];
	PRINT_FUNC;

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = CUDA_SET_DEVICE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argll = device;
	shmpkt->flags = CUDA_PKT_REQUEST;

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		;

	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDevice(int *device) {
	struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];
	PRINT_FUNC;

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = CUDA_GET_DEVICE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(struct cuda_packet);
	shmpkt->flags = CUDA_PKT_REQUEST;

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		;

	*device = *((int *)shmpkt->args[0].argull);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem  __dv(0), cudaStream_t stream  __dv(0)) {
	return lcudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	return lcudaSetupArgument(arg, size, offset);
}


cudaError_t cudaLaunch(const char *entry) {
	return lcudaLaunch(entry);
}

#if 0

cudaError_t cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
	struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = CUDA_CHOOSE_DEVICE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argp = (void *)device;
	shmpkt->args[1].argp = prop;
	shmpkt->flags = CUDA_PKT_REQUEST;

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		;

	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaThreadExit(void) {
	dlsym(RTLD_NEXT, __func__);
}

cudaError_t cudaThreadSynchronize(void) {
	return lcudaThreadSynchronize();
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

cudaError_t cudaGetLastError(void) {
	return lcudaGetLastError();
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

cudaError_t cudaMalloc(void **devPtr, size_t size) {
	return lcudaMalloc(devPtr, size);
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

cudaError_t cudaFree(void * devPtr) {
	return lcudaFree(devPtr);
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
	return lcudaMemcpy(dst, (const void *) src, count, kind);
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

	p_debug("symbol = %p, src = %p, count = %ld, offset = %ld, kind = %u\n",
			symbol, src, count, offset, kind);

	return lcudaMemcpyToSymbol(symbol, src, count, offset, kind);
}

cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol,
		size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind
		__dv(cudaMemcpyDeviceToHost)) {
	p_debug("dst = %p, symbol = %p (str %s), count = %ld, offset = %ld, kind = %d\n",
			dst, symbol, symbol, count, offset, kind);

	return lcudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
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
 * This call returns something different than cudaError_t so we must use
 * different something else than cudaErrorDL
 * @todo better handle DL error
 * @return empty cudaChannelFormatDesc if there is a problem with DL, as well
 * (maybe other NULL might means something else)
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

static void printFatBinary(void *fatC) {
	if (!fatC) {
		printd(DBG_ERROR, "NULL argument\n");
		return;
	}

	__cudaFatCudaBinary *fatbin = (__cudaFatCudaBinary *)fatC;

	printf("\tmagic=%lu\n", fatbin->magic);
	printf("\tversion=%lu\n", fatbin->version);
	printf("\tgpuInfoVersion=%lu\n", fatbin->gpuInfoVersion);

	printf("\tkey='%s'\n", fatbin->key);
	printf("\tident='%s'\n", fatbin->ident);
	printf("\tusageMode='%s'\n", fatbin->usageMode);

	printf("\tptx=%p\n", fatbin->ptx);
	if (fatbin->ptx) {
		printf("\tptx:\t\tgpuProfileName='%s'\n", fatbin->ptx->gpuProfileName);
		printf("\tptx:\t\tptx='%s'\n", fatbin->ptx->ptx);
	} else {
		printf("\tptx=%p\n", fatbin->ptx);
	}
}

void** __cudaRegisterFatBinary(void* cubin) {
	int err;
	volatile struct cuda_packet *shmpkt = (struct cuda_packet *)be_reg.shm[0];
	PRINT_FUNC;

	if (NEED_REGISTRATION) {
		err = registerWithBackend();
		if (err < 0) {
			fprintf(stderr, "Error registering with backend\n");
			return -1;
		}
		shmpkt = (struct cuda_packet *)be_reg.shm[0];
	}

	//printFatBinary(fatC);

	memset(shmpkt, 0, sizeof(struct cuda_packet));
	shmpkt->method_id = __CUDA_REGISTER_FAT_BINARY;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(struct cuda_packet); // offset
	shmpkt->flags = CUDA_PKT_REQUEST; // set this last FIXME sink spins on this

	wmb(); // flush writes from caches

	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();

	*count = *((int *)((void *)shmpkt + shmpkt->args[0].argull));
	return shmpkt->ret_ex_val.err;

fail:
	return NULL;
}
#endif

void __cudaUnregisterFatBinary(void** fatCubinHandle) {
	return;
}

#if 0
void l__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize) {
	typedef void** (* pFuncType)(void** fatCubinHandle, const char* hostFun,
			char* deviceFun, const char* deviceName, int thread_limit,
			uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);

	static pFuncType pFunc = NULL;

	l_printFuncSigImpl(__FUNCTION__);

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterFunction");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	(pFunc(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
			bid, bDim, gDim, wSize));
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize) {
	l__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
			deviceName, thread_limit, tid, bid,  bDim,  gDim, wSize);
}

void l__cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global) {
	typedef void** (* pFuncType)(void **fatCubinHandle, char *hostVar,
			char *deviceAddress, const char *deviceName, int ext, int vsize,
			int constant, int global);
	static pFuncType pFunc = NULL;

	l_printFuncSigImpl(__FUNCTION__);

	if (!pFunc) {
		pFunc = (pFuncType) dlsym(RTLD_NEXT, "__cudaRegisterVar");

		if (l_handleDlError() != 0)
			exit(-1);
	}

	(pFunc(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, vsize,
			constant, global));
}

/**
 * Andrew Kerr: "this function establishes a mapping between global variables
 * defined in .ptx or .cu modules and host-side variables. In PTX, global
 * variables have module scope and can be globally referenced by module and
 * variable name. In the CUDA Runtime API, globals in two modules must not have
 * the same name."
 */
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global) {
	l_printRegVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
			vsize, constant, global);

	l__cudaRegisterVar(fatCubinHandle, hostVar,
			deviceAddress, deviceName, ext, vsize, constant, global);
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
#endif


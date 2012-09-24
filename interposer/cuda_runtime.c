/**
 * @file cuda_runtime.c
 * @date Feb 27, 2011
 * @author Alex Merritt, merritt.alex@gatech.edu
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

// Note for reading the code in this file.
// 		It might be of use to enable 'syntax folding' in Vim
// 			:set foldmethod=syntax

// if this file is .c, if you do not define _GNU_SOURCE then it complains
// that RTLD_NEXT is undeclared
// if this file is  .cpp, then you get warning "_GNU_SOURCE" redefined
#define _GNU_SOURCE

/*-------------------------------------- INCLUDES ----------------------------*/

// System includes
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <uuid/uuid.h>

// CUDA includes
#include <__cudaFatFormat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

// Project includes
#include <assembly.h>
#include <cuda/bypass.h>
#include <cuda/hidden.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <cuda/packet.h> 
#include <debug.h>
#include <mq.h>
#include <util/compiler.h>
#include <util/x86_system.h>

// Directory-immediate includes
#include "timing.h"

/* preprocess out debug statements */
//#undef printd
//#define printd(level, fmt, args...)

#define USERMSG_PREFIX "=== INTERPOSER === "

/*-------------------------------------- EXTERNAL DEFINITIONS ----------------*/

/*-------------------------------------- INTERNAL STATE ----------------------*/

/*
 * CUDA State
 */

//! to indicate the error with the dynamic loaded library
//static cudaError_t cudaErrorDL = cudaErrorUnknown;

#if !(defined(TIMING) && defined(TIMING_NATIVE))
//! State machine for cudaGetLastError()
static cudaError_t cuda_err = cudaSuccess;
#endif

//! Reference count for register and unregister fatbinary invocations.
static unsigned int num_registered_cubins = 0;

/*
 * Scheduler state
 */

static bool scheduler_joined = false;
static struct mq_state recv_mq, send_mq;

/*
 * Assembly state
 */
static asmid_t assm_id;
static struct assembly_hint hint;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

/* forward delcaration into assembly/cuda_interface.c */
extern int assm_cuda_init(asmid_t id);
extern int assm_cuda_tini(void);

static int join_scheduler(void)
{
    int err;
    char uuid_str[64];
    assembly_key_uuid assm_key;

    if (scheduler_joined)
        return -1;

    scheduler_joined = true;

    memset(&recv_mq, 0, sizeof(recv_mq));
    memset(&send_mq, 0, sizeof(send_mq));

    /* initialize interfaces */
    err = attach_init(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_init: %d\n", err);
        fprintf(stderr, "Error attach_init: %d\n", err);
        return -1;
    }
	err = assembly_runtime_init(NODE_TYPE_MAPPER, NULL);
    if (err < 0) {
        printd(DBG_ERROR, "Error initializing assembly state\n");
        fprintf(stderr, "Error initializing assembly state\n");
        return -1;
    }

    /* tell daemon we wish to join */
    err = attach_send_connect(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_send_connect: %d\n", err);
        fprintf(stderr, "Error attach_send_connect: %d\n", err);
        return -1;
    }

    /* read the hint then send it to the daemon */
    err = assembly_read_hint(&hint);
    if (err < 0) {
        fprintf(stderr, "> Error reading hint file\n");
        return -1;
    }
    err = attach_send_request(&recv_mq, &send_mq, &hint, assm_key);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_send_request: %d\n", err);
        fprintf(stderr, "Error attach_send_request: %d\n", err);
        return -1;
    }
    uuid_unparse(assm_key, uuid_str);
    printd(DBG_INFO, "Importing assm key from scheduler: '%s'\n", uuid_str);

    /* 'fix' the assembly on the network */
    err = assembly_import(&assm_id, assm_key);
    BUG(assm_id == INVALID_ASSEMBLY_ID);
    if (err < 0) {
        printd(DBG_ERROR, "Error assembly_import: %d\n", err);
        fprintf(stderr, "Error assembly_import: %d\n", err);
        return -1;
    }
    assembly_print(assm_id);
    err = assembly_map(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error assembly_map: %d\n", err);
        fprintf(stderr, "Error assembly_map: %d\n", err);
        return -1;
    }

    /* XXX initialize the assembly/cuda_interface.c state AFTER the assembly
     * interface has been initialized AND an assembly has been imported */
    err = assm_cuda_init(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error initializing assembly cuda interface\n");
        fprintf(stderr, "Error initializing assembly cuda interface\n");
        return -1;
    }

    return 0;
}

static int leave_scheduler(void)
{
    int err;

    if (!scheduler_joined)
        return -1;

    scheduler_joined = false;

    err = attach_send_disconnect(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error telling daemon disconnect\n");
        fprintf(stderr, "Error telling daemon disconnect\n");
        return -1;
    }

    /* remove state associated with the assembly cuda interface */
    err = assm_cuda_tini();
    if (err < 0) {
        printd(DBG_ERROR, "Error deinitializing assembly cuda interface\n");
        fprintf(stderr, "Error deinitializing assembly cuda interface\n");
        return -1;
    }

    err = assembly_teardown(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error destroying assembly\n");
        fprintf(stderr, "Error destroying assembly\n");
        return -1;
    }

    err = assembly_runtime_shutdown();
    if (err < 0) {
        printd(DBG_ERROR, "Error cleaning up assembly runtime\n");
        fprintf(stderr, "Error cleaning up assembly runtime\n");
        return -1;
    }

    err = attach_tini(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error cleaning up attach state\n");
        fprintf(stderr, "Error cleaning up attach state\n");
        return -1;
    }

    return 0;
}

//! This appears in some values of arguments. I took this from
//! opt/cuda/include/cuda_runtime_api.h It looks as this comes from a default
//! value (dv)
#if !defined(__dv)
#	if defined(__cplusplus)
#		define __dv(v) \
        		= v
#	else
#		define __dv(v)
#	endif
#endif

/*-------------------------------------- INTERPOSING API ---------------------*/

//
// Thread Management API
//

cudaError_t cudaThreadExit(void)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadExit();
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaThreadExit(lat);
#endif

	update_latencies(lat, CUDA_THREAD_EXIT);
	return cerr;
}

cudaError_t cudaThreadSynchronize(void)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);
	
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadSynchronize();
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaThreadSynchronize(lat);
#endif

	update_latencies(lat, CUDA_THREAD_SYNCHRONIZE);
	return cerr;
}

//
// Error Handling API
//

const char* cudaGetErrorString(cudaError_t error)
{
#if defined(TIMING) && defined(TIMING_NATIVE)
	return bypass.cudaGetErrorString(error);
#else
	typedef const char*(*func_t)(cudaError_t);
	func_t func = (func_t)dlsym(RTLD_NEXT, __func__);
	if (func) return func(error);
	switch (error) {
		case cudaSuccess: return "cudaSuccess";
		case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
		case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
		case cudaErrorInitializationError: return "cudaErrorInitializationError";
		case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
		case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
		case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
		case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
		case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
		case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
		case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
		case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
		case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
		case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
		case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
		case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
		case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
		case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
		case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
		case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
		case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
		case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
		case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
		case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
		case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
		case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
		case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
		case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
		case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
		case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
		case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
		case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
		case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
		case cudaErrorNotReady: return "cudaErrorNotReady";
		case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";
		case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";
		case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";
		case cudaErrorNoDevice: return "cudaErrorNoDevice";
		case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";
		case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";
		case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";
		case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";
		case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";
		case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";
		case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";
		case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";
		case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";
		case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";
		case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";
		case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
		case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
		case cudaErrorUnknown:
		default:
			return "cudaErrorUnknown";
	}
#endif
}

cudaError_t cudaGetLastError(void)
{
#if defined(TIMING) && defined(TIMING_NATIVE)
	return bypass.cudaGetLastError();
#else
	return cuda_err; // ??
#endif
}

//
// Device Managment API
//

cudaError_t cudaGetDevice(int *device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "thread=%lu\n", pthread_self());
    LAT_DECLARE(lat);
	
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDevice(device);
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaGetDevice(device, lat);
#endif

	update_latencies(lat, CUDA_GET_DEVICE);
	return cerr;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDeviceCount(count);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaGetDeviceCount(count, lat);
	printd(DBG_DEBUG, "%d\n", *count);
#endif

	update_latencies(lat, CUDA_GET_DEVICE_COUNT);
	return cerr;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "dev=%d\n", device);
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDeviceProperties(prop,device);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + sizeof(*prop);
#else
	cerr = assm_cudaGetDeviceProperties(prop, device, lat);
#endif

	update_latencies(lat, CUDA_GET_DEVICE_PROPERTIES);
	return cerr;
}

cudaError_t cudaSetDevice(int device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "device=%d\n", device);
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetDevice(device);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaSetDevice(device, lat);
#endif

	update_latencies(lat, CUDA_SET_DEVICE);
	return cerr;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "flags=0x%x\n", flags);
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetDeviceFlags(flags);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaSetDeviceFlags(flags, lat);
#endif

	update_latencies(lat, CUDA_SET_DEVICE_FLAGS);
	return cerr;
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called; IGNORING\n");
    LAT_DECLARE(lat);

    /* XXX Ignore this stupid CUDA function */
    return cudaSuccess;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadExit();
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + (len * sizeof(*device_arr));
#else
	cerr = assm_cudaSetValidDevices(device_arr, len, lat);
#endif

	update_latencies(lat, CUDA_SET_VALID_DEVICES);
	return cerr;
}

//
// Stream Management API
//

// XXX NOTE XXX
//
// In v3.2 cudaStream_t is defined: typedef struct CUstream_st* cudaStream_t.
// So this is a pointer to a pointer allocated by the caller, and is meant as an
// output parameter. The compiler and debugger cannot tell me what this
// structure looks like, nor what its size is, so we'll have to maintain a cache
// of cudaStream_t in the sink processes, and return our own means of
// identification for them to the caller instead.  I'm guessing that the Runtime
// API allocates some kind of memory or something to this pointer internally,
// and uses this pointer as a key to look it up. We'll have to do the same.
//
// cuda v3.2 cudaStream_t: pointer to opaque struct
// cuda v2.3 cudaStream_t: integer

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaStreamCreate(pStream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaStreamCreate(pStream, lat);
#endif

	printd(DBG_DEBUG, "stream=%p\n", *pStream);

	update_latencies(lat, CUDA_STREAM_CREATE);
	return cerr;
}

#if 0
cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt;
	memset(&tpkt, 0, sizeof(tpkt));
	cerr = bypass.cudaStreamDestroy(stream);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt);
	shmpkt = &tpkt;
#else
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamDestroy(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_STREAM_DESTROY);
	return cerr;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt;
	memset(&tpkt, 0, sizeof(tpkt));
	cerr = bypass.cudaStreamQuery(stream);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt);
	shmpkt = &tpkt;
#else
	pack_cudaStreamQuery(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_STREAM_QUERY);
	return cerr;
}
#endif

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "stream=%p\n", stream);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaStreamSynchronize(stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaStreamSynchronize(stream, lat);
#endif

	update_latencies(lat, CUDA_STREAM_SYNCHRONIZE);
	return cerr;
}

//
// Execution Control API
//

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem  __dv(0), cudaStream_t stream  __dv(0)) {
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "grid={%d,%d,%d} block={%d,%d,%d} shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaConfigureCall(gridDim,blockDim,sharedMem,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaConfigureCall(gridDim, blockDim, sharedMem, stream, lat);
#endif

	update_latencies(lat, CUDA_CONFIGURE_CALL);
	return cerr;
}

#if 0
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(tsetup);

	printd(DBG_DEBUG, "func='%s'\n", func);

	if (!attr || !func) {
		return cudaErrorInvalidDeviceFunction;
	}

	TIMER_START(tsetup);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt;
	memset(&tpkt, 0, sizeof(tpkt));
	cerr = bypass.cudaFuncGetAttributes(attr,func);
	TIMER_END(tsetup, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt) + strlen(func) + 1;
	shmpkt = &tpkt;
#else
	void *func_shm = NULL; // Pointer to func str within shm
	void *attr_shm = NULL; // Pointer to attr within shm
	unsigned int func_len;
	TIMER_DECLARE1(twait);

	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	func_len = strlen(func) + 1;
	shmpkt->method_id = CUDA_FUNC_GET_ATTR;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset
	shmpkt->args[1].argull = (shmpkt->args[0].argull + sizeof(*attr)); // offset
	shmpkt->args[2].arr_argi[0] = func_len;
	func_shm = (void*)((uintptr_t)shmpkt + shmpkt->args[1].argull);
	memcpy(func_shm, func, func_len);
	shmpkt->len = sizeof(*shmpkt) + func_len;
	shmpkt->is_sync = true;
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	// Copy out the structure into the user argument
	attr_shm = (struct cudaFuncAttributes*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memcpy(attr, attr_shm, sizeof(*attr));
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_FUNC_GET_ATTR);
	return cerr;
}
#endif

cudaError_t cudaLaunch(const char *entry)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
	printd(DBG_DEBUG, "entry=%p\n", entry);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaLaunch(entry);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaLaunch(entry, lat);
#endif

	update_latencies(lat, CUDA_LAUNCH);
	return cerr;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "arg=%p size=%lu offset=%lu\n",
			arg, size, offset);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetupArgument(arg,size,offset);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + size;
#else
	cerr = assm_cudaSetupArgument(arg, size, offset, lat);
#endif

	update_latencies(lat, CUDA_SETUP_ARGUMENT);
	return cerr;
}

//
// Memory Management API
//

cudaError_t cudaFree(void * devPtr)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
	printd(DBG_DEBUG, "devPtr=%p\n", devPtr);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFree(devPtr);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaFree(devPtr, lat);
#endif

	update_latencies(lat, CUDA_FREE);
	return cerr;
}

cudaError_t cudaFreeArray(struct cudaArray * array)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
	printd(DBG_DEBUG, "array=%p\n", array);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFreeArray(array);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaFreeArray(array, lat);
#endif

	update_latencies(lat, CUDA_FREE_ARRAY);
	return cerr;
}

cudaError_t cudaFreeHost(void * ptr)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFreeHost(ptr);
	TIMER_END(t, lat->exec.call);
#else
	cerr = assm_cudaFreeHost(ptr, lat);
#endif

	update_latencies(lat, CUDA_FREE_HOST);
	return cerr;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaHostAlloc(pHost,size,flags);
	TIMER_END(t, lat->exec.call);
#else
	cerr = assm_cudaHostAlloc(pHost, size, flags, lat);
#endif

	printd(DBG_DEBUG, "host=%p size=%lu flags=0x%x (ignored)\n",
            *pHost, size, flags);
	update_latencies(lat, CUDA_HOST_ALLOC);
	return cerr;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMalloc(devPtr,size);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaMalloc(devPtr, size, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p size=%lu cerr=%d\n",
            *devPtr, size, cerr);
	update_latencies(lat, CUDA_MALLOC);
	return cerr;
}

cudaError_t cudaMallocArray(
		struct cudaArray **array, // stores a device address
		const struct cudaChannelFormatDesc *desc,
		size_t width, size_t height __dv(0), unsigned int flags __dv(0))
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMallocArray(array,desc,width,height,flags);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaMallocArray(array, desc, width, height, flags, lat);
#endif

	printd(DBG_DEBUG, "array=%p, desc=%p width=%lu height=%lu flags=0x%x\n",
			*array, desc, width, height, flags);
	update_latencies(lat, CUDA_MALLOC_ARRAY);
	return cerr;
}

cudaError_t cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height) {
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMallocPitch(devPtr,pitch,width,height);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaMallocPitch(devPtr, pitch, width, height, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p pitch=%lu\n", *devPtr, *pitch);
	update_latencies(lat, CUDA_MALLOC_PITCH);
	return cerr;
}

cudaError_t cudaMemcpy(void *dst, const void *src,
		size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d\n",
			dst, src, count, kind);

	method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToHost: id = CUDA_MEMCPY_H2H; break;
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_H2D; break;
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_D2D; break;
        default:
            fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
            abort();
	}

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpy(dst,src,count,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpy(dst, src, count, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d stream=%p\n",
			dst, src, count, kind, stream);

	method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToHost: id = CUDA_MEMCPY_ASYNC_H2H; break;
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_ASYNC_H2D; break;
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_ASYNC_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_ASYNC_D2D; break;
        default:
            fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
            abort();
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyAsync(dst,src,count,kind,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpyAsync(dst, src, count, kind, stream, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyFromSymbol(
		void *dst,
		const char *symbol, //! Either an addr of a var in the app, or a string
		size_t count, size_t offset __dv(0),
		enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "dst=%p symb=%p, count=%lu\n", dst, symbol, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_FROM_SYMBOL_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_FROM_SYMBOL_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyFromSymbol(dst,symbol,count,offset,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyToArray(
		struct cudaArray *dst,
		size_t wOffset, size_t hOffset,
		const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "dst=%p wOffset=%lu, hOffset=%lu, src=%p, count=%lu\n",
			dst, wOffset, hOffset, src, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_TO_ARRAY_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_TO_ARRAY_D2D; break;
		default: BUG(1);
	}

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToArray(dst,wOffset,hOffset,src,count,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}


cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset __dv(0),
		enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "symb=%p src=%p count=%lu\n", symbol, src, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_TO_SYMBOL_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_TO_SYMBOL_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToSymbol(symbol,src,count,offset,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpyToSymbol(symbol, src, count, offset, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyToSymbolAsync(
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "symb %p\n", symbol);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) + count;
#else
	cerr = assm_cudaMemcpyToSymbolAsync(symbol, src, count,
            offset, kind, stream, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemGetInfo(free,total);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaMemGetInfo(free, total, lat);
#endif

	printd(DBG_DEBUG, "free=%lu total=%lu\n", *free, *total);

	update_latencies(lat, CUDA_MEM_GET_INFO);
	return cerr;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemset(devPtr,value,count);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
	cerr = assm_cudaMemset(devPtr, value, count, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p value=%d count=%lu\n", devPtr, value, count);

	update_latencies(lat, CUDA_MEMSET);
	return cerr;
}

//
// Texture Management API
//

// see comments in __cudaRegisterTexture and cudaBindTextureToArray
cudaError_t cudaBindTexture(size_t *offset,
		const struct textureReference *texRef, //! addr of global var in app
		const void *devPtr,
		const struct cudaChannelFormatDesc *desc,
		size_t size __dv(UINT_MAX))
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaBindTexture(offset,texRef,devPtr,desc,size);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_BIND_TEXTURE);
	return cerr;
}

cudaError_t cudaBindTextureToArray(
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	cudaError_t cerr;

	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaBindTextureToArray(texRef,array,desc);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
	shmpkt = &tpkt;
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_BIND_TEXTURE_TO_ARRAY);
	return cerr;
}

struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
		enum cudaChannelFormatKind format)
{
#if 1
	// Doesn't need to be forwarded anywhere. Call the function in the NVIDIA
	// runtime, as this function just takes multiple variables and assigns them
	// to a struct. Why?
	const char *dl_err_str;
	typedef struct cudaChannelFormatDesc (*func)
		(int,int,int,int,enum cudaChannelFormatKind);
	func f = (func)dlsym(RTLD_NEXT, __func__);
	dl_err_str = dlerror();
	if (dl_err_str) {
		fprintf(stderr, USERMSG_PREFIX
				" error: could not find NVIDIA runtime: %s\n", dl_err_str);
		BUG(0);
	}
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u;"
			" passing to NV runtime\n",
			x, y, z, w, format);
	return f(x,y,z,w,format);
#else
	int err;
	err = join_scheduler(); // will return if already done
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		assert(0);
	}

	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u\n",
			x, y, z, w, format);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_CREATE_CHANNEL_DESC;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].arr_argii[0] = x;
	shmpkt->args[0].arr_argii[1] = y;
	shmpkt->args[0].arr_argii[2] = z;
	shmpkt->args[0].arr_argii[3] = w;
	shmpkt->args[1].arr_arguii[0] = format;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->args[0].desc;
#endif
}

#if 0
cudaError_t cudaGetTextureReference(
		const struct textureReference **texRef,
		const char *symbol)
{
	struct cuda_packet *shmpkt =
		(struct cuda_packet *)get_region(pthread_self());

	printd(DBG_DEBUG, "symbol=%p\n", symbol);

	BUG(0);

	return shmpkt->ret_ex_val.err;
}
#endif

//
// Version Management API
//

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaDriverGetVersion(driverVersion);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_DRIVER_GET_VERSION);
	return cerr;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaRuntimeGetVersion(runtimeVersion);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_RUNTIME_GET_VERSION);
	return cerr;
}

//
// Undocumented API
//

void** __cudaRegisterFatBinary(void* cubin)
{
	void **handle;
	//struct cuda_packet *shmpkt;
    LAT_DECLARE(lat);

	if (num_registered_cubins <= 0) {
        fill_bypass(&bypass);
#if defined(TIMING) && defined(TIMING_NATIVE)
        /* nothing */
#else
		if (0 > join_scheduler()) {
			fprintf(stderr, "Error attaching to assembly runtime\n");
			assert(0);
		}
#endif
	}

	num_registered_cubins++;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	handle = bypass.__cudaRegisterFatBinary(cubin);
	TIMER_END(t, lat->exec.call);
	cache_num_entries_t _notused;
	lat->len = sizeof(tpkt) + getFatRecPktSize(cubin, &_notused);
#else
    handle = assm__cudaRegisterFatBinary(cubin, lat);
	printd(DBG_DEBUG, "handle=%p\n", handle);
#endif

	update_latencies(lat, __CUDA_REGISTER_FAT_BINARY);
	return handle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle)
{
    LAT_DECLARE(lat);

	printd(DBG_INFO, "handle=%p\n", fatCubinHandle);

	num_registered_cubins--;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaUnregisterFatBinary(fatCubinHandle);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt);
#else
    assm__cudaUnregisterFatBinary(fatCubinHandle, lat);
#endif

	update_latencies(lat, __CUDA_UNREGISTER_FAT_BINARY);

	if (num_registered_cubins <= 0) { // only detach on last unregister
#if !(defined(TIMING) && defined(TIMING_NATIVE))
		leave_scheduler();
#endif
		print_latencies();
	}

	return;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
    LAT_DECLARE(lat);

	printd(DBG_DEBUG, "handle=%p hostFun=%p deviceFun=%s deviceName=%s\n",
			fatCubinHandle, hostFun, deviceFun, deviceName);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaRegisterFunction(fatCubinHandle,hostFun,deviceFun,deviceName,
			thread_limit,tid,bid,bDim,gDim,wSize);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) +
		getSize_regFuncArgs(fatCubinHandle, hostFun, deviceFun, deviceName,
				thread_limit, tid, bid, bDim, gDim, wSize);
#else
    assm__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
            deviceName, thread_limit, tid, bid, bDim, gDim, wSize, lat);
#endif

	update_latencies(lat, __CUDA_REGISTER_FUNCTION);
	return;
}

void __cudaRegisterVar(
		void **fatCubinHandle,	//! cubin this var associates with
		char *hostVar,			//! addr of a var within app (not string)
		char *deviceAddress,	//! 8-byte device addr
		const char *deviceName, //! actual string
		int ext, int vsize, int constant, int global)
{
    LAT_DECLARE(lat);
	printd(DBG_DEBUG, "symbol=%p\n", hostVar);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaRegisterVar(fatCubinHandle,hostVar,deviceAddress,deviceName,
			ext,vsize,constant,global);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(tpkt) +
		getSize_regVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
				ext, vsize, constant, global);
#else
	assm__cudaRegisterVar(fatCubinHandle,hostVar,deviceAddress,deviceName,
			ext,vsize,constant,global, lat);
#endif

	update_latencies(lat, __CUDA_REGISTER_VARIABLE);
	return;
}

#if 0
// first three args we treat as handles (i.e. only pass down the pointer
// addresses)
void __cudaRegisterTexture(
		void** fatCubinHandle,
		//! address of a global variable within the application; store the addr
		const struct textureReference* texRef,
		//! 8-byte device address; dereference it once to get the address
		const void** deviceAddress,
		const char* texName, //! actual string
		int dim, int norm, int ext)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "handle=%p texRef=%p devAddr=%p *devAddr=%p texName=%s"
			" dim=%d norm=%d ext=%d\n",
			fatCubinHandle, texRef, deviceAddress, *deviceAddress, texName,
			dim, norm, ext);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt; // XXX this might add a lot to the stack
	memset(&tpkt, 0, sizeof(tpkt));
	bypass.__cudaRegisterTexture(fatCubinHandle,texRef,deviceAddress,texName,
			dim,norm,ext);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt) + strlen(texName) + 1;
	shmpkt = &tpkt;
#else
	uintptr_t shm_ptr;
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	shm_ptr = (uintptr_t)shmpkt;
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = __CUDA_REGISTER_TEXTURE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argdp = fatCubinHandle; // pointer copy
	// Subsequent calls to texture functions will provide the address of this
	// global as a parameter; there we will copy the data in the global into the
	// cuda_packet, copy it to the variable registered in the sink process, and
	// invoke such functions with the variable address allocated in the sink. We
	// still need this address, because we cannot tell the application to change
	// which address it provides us---we have to perform lookups of the global's
	// address in the application with what we cache in the sink process.
	shmpkt->args[1].argp = (void*)texRef; // pointer copy
	shmpkt->args[2].texRef = *texRef; // state copy
	shmpkt->args[3].argp = (void*)*deviceAddress; // copy actual address
	shmpkt->args[4].argull = sizeof(*shmpkt); // offset
	shm_ptr += sizeof(*shmpkt);
	memcpy((void*)shm_ptr, texName, strlen(texName) + 1);
	shmpkt->args[5].arr_argii[0] = dim;
	shmpkt->args[5].arr_argii[1] = norm;
	shmpkt->args[5].arr_argii[2] = ext;
	shmpkt->len = sizeof(*shmpkt) + strlen(texName) + 1;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
#endif

	update_latencies(lat, __CUDA_REGISTER_TEXTURE);
	return;
}
#endif

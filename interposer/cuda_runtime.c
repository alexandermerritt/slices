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

// CUDA includes
#include <__cudaFatFormat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

// Other project includes
#include <shmgrp.h>

// Project includes
#include <assembly.h>
#include <cuda/hidden.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <cuda/packet.h> 
#include <debug.h>
#include <util/compiler.h>
#include <util/x86_system.h>

// Directory-immediate includes
#include "timing.h"

/* preprocess out debug statements */
//#undef printd
//#define printd(level, fmt, args...)

#define USERMSG_PREFIX "=== INTERPOSER === "

// Functions from ./shm.c
extern int attach_assembly_runtime(void);
extern void detach_assembly_runtime(void);
extern void* get_region(pthread_t tid);

/*-------------------------------------- INTERNAL STATE ----------------------*/

//! to indicate the error with the dynamic loaded library
//static cudaError_t cudaErrorDL = cudaErrorUnknown;

//! State machine for cudaGetLastError()
static cudaError_t cuda_err = cudaSuccess;

//! Reference count for register and unregister fatbinary invocations.
static unsigned int num_registered_cubins = 0;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

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

/** XXX Maybe this should be defined in some common 'glue' code? */
#define HANDOFF_AND_SPIN(_pkt) \
{ \
	wmb(); /* make writes visible */ \
	_pkt->flags |= CUDA_PKT_REQUEST; \
	while (!(_pkt->flags & CUDA_PKT_RESPONSE)) \
		rmb(); \
}

/*-------------------------------------- INTERPOSING API ---------------------*/

// TODO We could support a circular queue within each shm region in the future.

//
// Thread Management API
//

cudaError_t cudaThreadExit(void)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "called\n");
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaThreadExit(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaThreadSynchronize(void)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "called\n");
	TIMER_DECLARE1(t);
	
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaThreadSynchronize(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

//
// Error Handling API
//

const char* cudaGetErrorString(cudaError_t error)
{
	// Doesn't need to be forwarded anywhere. Call the function in the NVIDIA
	// runtime (FIXME an assumption we maybe shouldn't make?)
	const char *dl_err_str;
	typedef const char * (*func)(cudaError_t err);
	func f = (func)dlsym(RTLD_NEXT, "cudaGetErrorString");
	dl_err_str = dlerror();
	if (dl_err_str) {
		fprintf(stderr, "Error: No NVIDIA runtime exists: %s\n", dl_err_str);
		return NULL;
	}
	return f(error);
}

cudaError_t cudaGetLastError(void)
{
	return cuda_err; // ??
}

//
// Device Managment API
//

cudaError_t cudaGetDevice(int *device)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "thread=%lu\n", pthread_self());
	TIMER_DECLARE1(t);
	
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaGetDevice(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaGetDevice(shmpkt, device);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaGetDeviceCount(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaGetDeviceCount(shmpkt, count);
	printd(DBG_DEBUG, "%d\n", *count);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "dev=%d\n", device);
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaGetDeviceProperties(shmpkt, device);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaGetDeviceProperties(shmpkt, (shmpkt+sizeof(*shmpkt)), prop);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetDevice(int device)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "device=%d\n", device);
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaSetDevice(shmpkt, device);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "flags=0x%x\n", flags);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaSetDeviceFlags(shmpkt, flags);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "called\n");

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaSetValidDevices(shmpkt, (shmpkt + sizeof(*shmpkt)),
			device_arr, len);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
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
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamCreate(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaStreamCreate(shmpkt, pStream);
	printd(DBG_DEBUG, "stream=%p\n", *pStream);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamDestroy(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamQuery(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamSynchronize(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

//
// Execution Control API
//

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem  __dv(0), cudaStream_t stream  __dv(0)) {
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "grid={%d,%d,%d} block={%d,%d,%d} shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaConfigureCall(shmpkt, gridDim, blockDim, sharedMem, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
	struct cuda_packet *shmpkt;
	void *func_shm = NULL; // Pointer to func str within shm
	void *attr_shm = NULL; // Pointer to attr within shm
	unsigned int func_len;

	printd(DBG_DEBUG, "func='%s'\n", func);

	if (!attr || !func) {
		return cudaErrorInvalidDeviceFunction;
	}

	TIMER_DECLARE2(tsetup, twait);
	TIMER_START(tsetup);
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
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	// Copy out the structure into the user argument
	attr_shm = (struct cudaFuncAttributes*)
		((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memcpy(attr, attr_shm, sizeof(*attr));
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaLaunch(const char *entry)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "entry=%p\n", entry);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaLaunch(shmpkt, entry);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "arg=%p size=%lu offset=%lu\n",
			arg, size, offset);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaSetupArgument(shmpkt, (shmpkt + sizeof(*shmpkt)),
			arg, size, offset);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

//
// Memory Management API
//

cudaError_t cudaFree(void * devPtr)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "devPtr=%p\n", devPtr);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaFree(shmpkt, devPtr);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaFreeArray(struct cudaArray * array)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "array=%p\n", array);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaFreeArray(shmpkt, array);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaFreeHost(void * ptr)
{
	if (ptr)
		free(ptr);
	return cudaSuccess;

#if 0 // Working code that forwards the RPC to the assembly runtime.
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "ptr=%p\n", ptr);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_FREE_HOST;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argp = ptr;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
#endif
}

/**
 * This function, according to the NVIDIA CUDA API specifications, seems to just
 * be a combined malloc+mlock that is additionally made visible to the CUDA
 * runtime and NVIDIA driver to optimize memory movements carried out in
 * subsequent calls to cudaMemcpy. Since we currently assume another process
 * carries out our RPC requests (local- or remotesink), there's no point in
 * forwarding this function: cudaHostAlloc returns an application virtual
 * address, it is useless here.
 *
 * We ignore flags for now, it only specifies performance not correctness.
 */
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	void *pinned_mem = NULL;
	int err = 0;

	TIMER_DECLARE1(t);
	TIMER_START(t);
#if TIMING
	struct rpc_latencies lat;
	memset(&lat, 0, sizeof(lat));
#endif

	pinned_mem = malloc(size);
	if (!pinned_mem) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		return cudaErrorMemoryAllocation;
	}

	// if this fails, nothing will break
	err = mlock(pinned_mem, size);
	if (err < 0) {
		printd(DBG_WARNING, "memory pinning failed: %s\n", strerror(errno));
	}

	TIMER_END(t, lat.lib.setup);
	update_latencies(&lat);

	*pHost = pinned_mem;
	printd(DBG_DEBUG, "host=%p size=%lu flags=0x%x (ignored)\n", *pHost, size, flags);

	return cudaSuccess;

#if 0 // Working code that forwards the RPC to the assembly runtime.
	struct cuda_packet *shmpkt;

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_HOST_ALLOC;
	shmpkt->thr_id = pthread_self();
	// We'll expect the value of *pHost to reside in args[0].argull
	shmpkt->args[1].arr_argi[0] = size;
	shmpkt->args[2].argull = flags;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	*pHost = (void*)shmpkt->args[0].argull;
	printd(DBG_DEBUG, "host=%p size=%lu flags=0x%x\n", *pHost, size, flags);
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
#endif
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMalloc(shmpkt, size);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaMalloc(shmpkt, devPtr);
	printd(DBG_DEBUG, "devPtr=%p size=%lu\n", *devPtr, size);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMallocArray(
		struct cudaArray **array, // stores a device address
		const struct cudaChannelFormatDesc *desc,
		size_t width, size_t height __dv(0), unsigned int flags __dv(0))
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMallocArray(shmpkt, desc, width, height, flags);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaMallocArray(shmpkt, array);
	printd(DBG_DEBUG, "array=%p, desc=%p width=%lu height=%lu flags=0x%x\n",
			*array, desc, width, height, flags);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height) {
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMallocPitch(shmpkt, width, height);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaMallocPitch(shmpkt, devPtr, pitch);
	printd(DBG_DEBUG, "devPtr=%p pitch=%lu\n", *devPtr, *pitch);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpy(void *dst, const void *src,
		size_t count, enum cudaMemcpyKind kind)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE2(tsetup, twait);

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d\n",
			dst, src, count, kind);

	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpy(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, src, count, kind);
	TIMER_PAUSE(tsetup);

	if (kind == cudaMemcpyHostToHost)
		return cudaSuccess;

	TIMER_START(twait);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	extract_cudaMemcpy(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, src, count, kind);
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE2(tsetup,twait);

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d stream=%p\n",
			dst, src, count, kind, stream);

	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpyAsync(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, src, count, kind, stream);
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	extract_cudaMemcpyAsync(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, src, count, kind, stream);
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyFromSymbol(
		void *dst,
		const char *symbol, //! Either an addr of a var in the app, or a string
		size_t count, size_t offset __dv(0),
		enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE2(tsetup, twait);

	printd(DBG_DEBUG, "dst=%p symb=%p, count=%lu\n", dst, symbol, count);

	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpyFromSymbol(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, symbol, count, offset, kind);
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	extract_cudaMemcpyFromSymbol(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, symbol, count, offset, kind);
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyToArray(
		struct cudaArray *dst,
		size_t wOffset, size_t hOffset,
		const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "dst=%p wOffset=%lu, hOffset=%lu, src=%p, count=%lu\n",
			dst, wOffset, hOffset, src, count);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpyToArray(shmpkt, (shmpkt + sizeof(*shmpkt)),
			dst, wOffset, hOffset, src, count, kind);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}


cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset __dv(0),
		enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "symb=%p src=%p count=%lu\n", symbol, src, count);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpyToSymbol(shmpkt, (shmpkt + sizeof(*shmpkt)),
			symbol, src, count, offset, kind);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyToSymbolAsync(
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "symb %p\n", symbol);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemcpyToSymbolAsync(shmpkt, (shmpkt + sizeof(*shmpkt)),
			symbol, src, count, offset, kind, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	struct cuda_packet *shmpkt;

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_MEM_GET_INFO;
	shmpkt->thr_id = pthread_self();
	// We expect to read the values for free and total in
	// 		args[0].arr_argi[0], and
	// 		args[0].arr_argi[1]
	// respectively.
	shmpkt->len = sizeof(*shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	*free = shmpkt->args[0].arr_argi[0];
	*total = shmpkt->args[0].arr_argi[1];
	printd(DBG_DEBUG, "free=%lu total=%lu\n", *free, *total);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaMemset(shmpkt, devPtr, value, count);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	printd(DBG_DEBUG, "devPtr=%p value=%d count=%lu\n", devPtr, value, count);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
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
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);
	printd(DBG_DEBUG, "called\n");

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaBindTexture(shmpkt, texRef, devPtr, desc, size);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaBindTexture(shmpkt, offset);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaBindTextureToArray(
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "called\n");
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaBindTextureToArray(shmpkt, texRef, array, desc);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
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
	err = attach_assembly_runtime(); // will return if already done
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
	HANDOFF_AND_SPIN(shmpkt);
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
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaDriverGetVersion(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaDriverGetVersion(shmpkt, driverVersion);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaRuntimeGetVersion(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaRuntimeGetVersion(shmpkt, runtimeVersion);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

//
// Undocumented API
//

void** __cudaRegisterFatBinary(void* cubin)
{
	int err;
	void **handle;
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	num_registered_cubins++;

	err = attach_assembly_runtime(); // will return if already done
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		assert(0);
	}

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaRegisterFatBinary(shmpkt, (shmpkt + sizeof(*shmpkt)), cubin);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	extract_cudaRegisterFatBinary(shmpkt, &handle);
	printd(DBG_DEBUG, "handle=%p\n", shmpkt->ret_ex_val.handle);

	update_latencies(&shmpkt->lat);
	return handle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);
	printd(DBG_INFO, "handle=%p\n", fatCubinHandle);
	num_registered_cubins--;

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaUnregisterFatBinary(shmpkt, fatCubinHandle);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);

	if (num_registered_cubins == 0) { // only detach on last unregister
		detach_assembly_runtime();
		print_latencies();
	}

	return;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "handle=%p hostFun=%p deviceFun=%s deviceName=%s\n",
			fatCubinHandle, hostFun, deviceFun, deviceName);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaRegisterFunction(shmpkt, (shmpkt + sizeof(*shmpkt)),
			fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
			bid, bDim, gDim, wSize);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return;
}

void __cudaRegisterVar(
		void **fatCubinHandle,	//! cubin this var associates with
		char *hostVar,			//! addr of a var within app (not string)
		char *deviceAddress,	//! 8-byte device addr
		const char *deviceName, //! actual string
		int ext, int vsize, int constant, int global)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);
	printd(DBG_DEBUG, "symbol=%p\n", hostVar);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaRegisterVar(shmpkt, (shmpkt + sizeof(*shmpkt)),
			fatCubinHandle, hostVar, deviceAddress, deviceName, ext, vsize,
			constant, global);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return;
}

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
	void *shm_ptr;

	printd(DBG_DEBUG, "handle=%p texRef=%p devAddr=%p *devAddr=%p texName=%s"
			" dim=%d norm=%d ext=%d\n",
			fatCubinHandle, texRef, deviceAddress, *deviceAddress, texName,
			dim, norm, ext);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shm_ptr = shmpkt = (struct cuda_packet *)get_region(pthread_self());
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
	memcpy(shm_ptr, texName, strlen(texName) + 1);
	shmpkt->args[5].arr_argii[0] = dim;
	shmpkt->args[5].arr_argii[1] = norm;
	shmpkt->args[5].arr_argii[2] = ext;
	shmpkt->len = sizeof(*shmpkt) + strlen(texName) + 1;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	HANDOFF_AND_SPIN(shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return;
}

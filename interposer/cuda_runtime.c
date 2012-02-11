/**
 * @file libci.c
 *
 * @date Feb 27, 2011
 * @author Magda Slawinska, magg@gatech.edu
 * @author Alex Merritt, merritt.alex@gatech.edu
 *
 * @brief Interposes the cuda calls and prints arguments of the call. It
 * supports the 3.2 CUDA Toolkit, specifically
 *   CUDA Runtime API Version 3.2
 *   #define CUDART_VERSION  3020
 *
 * To prepare the file I processed the
 * /opt/cuda/include/cuda_runtime_api_no_comments.h and removed the comments. I
 * also removed CUDARTAPI and __host__ modifiers.  Then I have a list of
 * function signatures that I need to interpose.  You can see the script in
 * cuda_rt_api.
 *
 * 2011-02-09 It looks that  in my library currently is  95 calls plus 6 calls
 * undocumented.
 * @todo Write a script that checks if the number or api are identical in
 * cuda_runtime_api.h
 * and in our file.
 *
 *
 * @todo There is one thing: the prototypes or signatures of CUDA functions have
 * modifiers CUDARTAPI which is __stdcall and __host__ which is
 * __location__(host) as defined in file /opt/cuda/include/host_defines.h The
 * question is if this has any impact on the interposed calls. I guess not.  But
 * I might be wrong.
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
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* preprocess out debug statements */
//#undef printd
//#define printd(level, fmt, args...)

#define USERMSG_PREFIX "=== INTERPOSER === "

/*-------------------------------------- INTERNAL STATE ----------------------*/

//! to indicate the error with the dynamic loaded library
//static cudaError_t cudaErrorDL = cudaErrorUnknown;

//! State machine for cudaGetLastError()
static cudaError_t cuda_err = cudaSuccess;

static struct shm_regions *cuda_regions;

//! Reference count for register and unregister fatbinary invocations.
static unsigned int num_registered_cubins = 0;

#define MAX_REGISTERED_VARS 5210
//! Symbol addresses from __cudaRegisterVar. Used to determine if the symbol
//! parameter in certain functions is actually the address of a variable, or
//! the string name of one of the variables in functions which accept symbols.
//! TODO Make this cleaner code.
static uintptr_t registered_vars[MAX_REGISTERED_VARS];
static unsigned int num_registered_vars = 0;

/*-------------------------------------- TIMER DEFINITIONS -------------------*/
#ifdef TIMING

// Timer resolution set by TIMER_ACCURACY in util/timer.h

static struct timer attach_timer; //! Timer used to measure joining the runtime
static uint64_t attach = 0UL; //! Time spent joining the runtime

static struct timer detach_timer; //! Timer used to measure departing the runtime
static uint64_t detach = 0UL; //! Time spent departing the runtime

static struct rpc_latencies lat; //! Aggregate latencies over all RPCs

/** Initialize global timers above. */
static void
timers_init(void)
{
	timer_init(CLOCK_REALTIME, &attach_timer);
	timer_init(CLOCK_REALTIME, &detach_timer);
}

static inline void
update_latencies(const struct rpc_latencies *l)
{
	lat.lib.setup += l->lib.setup;
	lat.lib.wait += l->lib.wait;

	lat.exec.setup += l->exec.setup;
	lat.exec.call += l->exec.call;

	lat.rpc.append += l->rpc.append;
	lat.rpc.send += l->rpc.send;
	lat.rpc.wait += l->rpc.wait;
	lat.rpc.recv += l->rpc.recv;

	lat.remote.batch_exec += l->remote.batch_exec;
}

static inline void
print_latencies(void)
{
	printf(USERMSG_PREFIX TIMERMSG_PREFIX
			"lib.setup %lu lib.wait %lu "
			"exec.setup %lu exec.call %lu "
			"rpc.append %lu rpc.send %lu rpc.wait %lu rpc.recv %lu "
			"remote.batch_exec %lu"
			"\n",
			lat.lib.setup, lat.lib.wait,
			lat.exec.setup, lat.exec.call,
			lat.rpc.append, lat.rpc.send, lat.rpc.wait, lat.rpc.recv,
			lat.remote.batch_exec);
}

#else	/* !TIMING */

#define update_latencies(lat)
#define print_latencies()

#endif	/* TIMING */

/*-------------------------------------- SHMGRP DEFINITIONS ------------------*/

//! Amount of memory to allocate for each CUDA thread, in bytes.
//! XXX FIXME Put in checks to make sure the usercode isn't trying to copy more
//! than we allocate.
#define THREAD_SHM_SIZE					(512 << 20)

/**
 * Region of memory we've mapped in with the runtime. Each region is assigned to
 * a unique thread in the CUDA application.
 */
struct shm_region {
	struct list_head link;
	struct shmgrp_region shmgrp_region;
	pthread_t tid;	//! Application thread assigned this region
};

//! List of shm regions maintained with the assembly runtime.
struct shm_regions {
	struct list_head list;
	pthread_mutex_t lock; //! two-fold lock: this structure and the shmgrp API
};

//! Iterator loop for shm regions.
#define __shm_for_each_region(regions, region)	\
	list_for_each_entry(region, &((regions)->list), link)

//! Iterator loop for shm regions, allowing us to remove regions as we iterate.
#define __shm_for_each_region_safe(regions, region, tmp)	\
	list_for_each_entry_safe(region, tmp, &((regions)->list), link)

// All the __shm* functions should not be called directly. They should only be
// called by the other functions just below them, as any locking that needs to
// happen is handled correctly within those.

static inline struct shm_region *
__shm_get_region(struct shm_regions *regions, pthread_t tid)
{
	struct shm_region *region;
	__shm_for_each_region(regions, region)
		if (pthread_equal(region->tid, tid) != 0)
			return region;
	return NULL;
}

static inline void
__shm_add_region(struct shm_regions *regions, struct shm_region *region)
{
	list_add(&region->link, &regions->list);
}

static inline void
__shm_rm_region(struct shm_region *region)
{
	list_del(&region->link);
}

static inline bool
__shm_has_regions(struct shm_regions *regions)
{
	return !(list_empty(&regions->list));
}

// Should be called within the first CUDA call interposed. It is okay to call
// this function more than once, it will only have an effect the first time.
static int attach_assembly_runtime(void)
{
	int err;
	if (cuda_regions)
		return 0;
#ifdef TIMING
	timers_init();
	timer_start(&attach_timer);
#endif
	cuda_regions = calloc(1, sizeof(*cuda_regions));
	if (!cuda_regions) {
		fprintf(stderr, "Out of memory\n");
		return -1;
	}
	INIT_LIST_HEAD(&cuda_regions->list);
	err = shmgrp_init();
	if (err < 0) {
		fprintf(stderr, "Error initializing shmgrp state\n");
		return -1;
	}
	err = shmgrp_join(ASSEMBLY_SHMGRP_KEY);
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		return -1;
	}
#ifdef TIMING
	attach = timer_end(&attach_timer, TIMER_ACCURACY);
	printf(USERMSG_PREFIX TIMERMSG_PREFIX "attach %lu\n", attach);
#endif
	return 0;
}

// Should be called within the last CUDA call interposed. It is okay to call
// this function more than once, it will only have an effect the first time.
static void detach_assembly_runtime(void)
{
	int err;
	struct shm_region *region, *tmp;
	if (!cuda_regions)
		return;
#ifdef TIMING
	timer_start(&detach_timer);
#endif
	if (__shm_has_regions(cuda_regions)) {
		__shm_for_each_region_safe(cuda_regions, region, tmp) {
			__shm_rm_region(region);
			err = shmgrp_rmreg(ASSEMBLY_SHMGRP_KEY, region->shmgrp_region.id);
			if (err < 0) {
				printd(DBG_ERROR, "Error destroying region %d\n",
						region->shmgrp_region.id);
			}
			free(region);
		}
	}
	free(cuda_regions);
	err = shmgrp_leave(ASSEMBLY_SHMGRP_KEY);
	if (err < 0)
		fprintf(stderr, "Error detaching from assembly runtime\n");
	err = shmgrp_tini();
	if (err < 0)
		fprintf(stderr, "Error uninitializing shmgrp state\n");
#ifdef TIMING
	detach = timer_end(&detach_timer, TIMER_ACCURACY);
	printf(USERMSG_PREFIX TIMERMSG_PREFIX "detach %lu\n", detach);
#endif
}

// Create a new shared memory region with the runtime, and add a new entry to
// the region list for us. This function must be thread-safe (newly detected
// threads will ask for a new region, and will call this). If the tid has
// already been allocated a memory region, we simply return that mapping.
static void * __add_shm(size_t size, pthread_t tid)
{
	int err;
	struct shm_region *region;
	shmgrp_region_id id;
	region = __shm_get_region(cuda_regions, tid);
	if (region)
		return region->shmgrp_region.addr;
	region = calloc(1, sizeof(*region));
	if (!region) {
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&region->link);
	region->tid = tid;
	err = shmgrp_mkreg(ASSEMBLY_SHMGRP_KEY, size, &id);
	if (err < 0) {
		fprintf(stderr, "Error creating a new memory region with assembly\n");
		goto fail;
	}
	err = shmgrp_leader_region(ASSEMBLY_SHMGRP_KEY, id, &region->shmgrp_region);
	if (err < 0) {
		fprintf(stderr, "Error accessing existing shm region %d\n", id);
		goto fail;
	}
	__shm_add_region(cuda_regions, region);
	return region->shmgrp_region.addr;
fail:
	if (region)
		free(region);
	return NULL;
}

//! Each interposed call will invoke this to locate the shared memory region
//! allocated to the thread calling it. If one does not exist, one is allocated.
//! Thus if there are no errors, this function will always return a valid
//! address.
static inline void *
get_region(pthread_t tid)
{
	void *addr;
	pthread_mutex_lock(&cuda_regions->lock);
	addr = __add_shm(THREAD_SHM_SIZE, tid);
	pthread_mutex_unlock(&cuda_regions->lock);
	return addr;
}

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

// Lookup 'symbol' in our list of known registered variable addresses. If it
// exists, then store it as the address of a variable residing in application
// space. Else, it must be a string literal naming a global variable. If the
// latter, copy the string to the shm region and indicate the symbol is a string
// by setting a flag in the packet.
static inline bool __func_symb_param_is_string(const char *symbol)
{
	unsigned int symb = 0;
	while (symb < num_registered_vars)
		if (registered_vars[symb++] == (uintptr_t)symbol)
			return false;
	return true;
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_THREAD_EXIT;
	shmpkt->thr_id = pthread_self();
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_THREAD_SYNCHRONIZE;
	shmpkt->thr_id = pthread_self();
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_GET_DEVICE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // FIXME just use an arg
	shmpkt->len = sizeof(*shmpkt) + sizeof(int);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*device = *((int*)((uintptr_t)shmpkt + shmpkt->args[0].argull));
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	int err;
	struct cuda_packet *shmpkt;

	err = attach_assembly_runtime();
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		assert(0);
	}

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());

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

	memset((void*)shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_GET_DEVICE_COUNT;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset; FIXME just use an arg
	shmpkt->len = sizeof(*shmpkt) + sizeof(int);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST; // set this last FIXME sink spins on this
	wmb(); // flush writes from caches
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*count = *((int *)((uintptr_t)shmpkt + shmpkt->args[0].argull));
	printd(DBG_DEBUG, "%d\n", *count);
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	struct cuda_packet *shmpkt;
	struct cudaDeviceProp *prop_shm = NULL;

	printd(DBG_DEBUG, "dev=%d\n", device);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_GET_DEVICE_PROPERTIES;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset
	shmpkt->args[1].argll = device;
	shmpkt->len = sizeof(*shmpkt) + sizeof(*prop);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	prop_shm = (struct cudaDeviceProp*)
					((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memcpy(prop, prop_shm, sizeof(struct cudaDeviceProp));

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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_SET_DEVICE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argll = device;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_SET_DEVICE_FLAGS;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = flags;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;
	printd(DBG_DEBUG, "called\n");

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shm_ptr = shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_SET_VALID_DEVICES;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt);
	shmpkt->args[1].argll = len;
	shm_ptr += shmpkt->args[0].argull;
	memcpy(shm_ptr, device_arr, (len * sizeof(*device_arr)));
	shmpkt->len = sizeof(*shmpkt) + (len * sizeof(*device_arr));
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_STREAM_CREATE;
	shmpkt->thr_id = pthread_self();
	// We'll expect the value of *pStream to exist in args[0]
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*pStream = shmpkt->args[0].stream;
	printd(DBG_DEBUG, "stream=%p\n", *pStream);
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_STREAM_SYNCHRONIZE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].stream = stream;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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

	printd(DBG_DEBUG, "grid={%d,%d,%d} block={%d,%d,%d} shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_CONFIGURE_CALL;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].arg_dim = gridDim; // = on structs works :)
	shmpkt->args[1].arg_dim = blockDim;
	shmpkt->args[2].arr_argi[0] = sharedMem;
	shmpkt->args[3].argull = (uint64_t)stream;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = false;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	printd(DBG_DEBUG, "ret err = %d\n", shmpkt->ret_ex_val.err);
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_LAUNCH;
	shmpkt->thr_id = pthread_self();
	// FIXME We assume entry is just a memory pointer, not a string.
	shmpkt->args[0].argull = (uintptr_t)entry;
	printd(DBG_DEBUG, "entry=%p\n", (void*)shmpkt->args[0].argull);
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = false;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "arg=%p size=%lu offset=%lu\n",
			arg, size, offset);

	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_SETUP_ARGUMENT;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt);
	shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memcpy(shm_ptr, arg, size);
	shmpkt->len = sizeof(*shmpkt) + size;
	shmpkt->is_sync = false;
	shmpkt->args[1].arr_argi[0] = size;
	shmpkt->args[1].arr_argi[1] = offset;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_FREE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argp = devPtr;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_FREE_ARRAY;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].cudaArray = array;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaFreeHost(void * ptr)
{
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*pHost = (void*)shmpkt->args[0].argull;
	printd(DBG_DEBUG, "host=%p size=%lu flags=0x%x\n", *pHost, size, flags);
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	struct cuda_packet *shmpkt;

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_MALLOC;
	shmpkt->thr_id = pthread_self();
	// We expect the sink to write the value of devPtr to args[0].argull
	shmpkt->args[1].arr_argi[0] = size;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*devPtr = (void*)shmpkt->args[0].argull;
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
	void *shm_ptr;

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shm_ptr = shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_MALLOC_ARRAY;
	shmpkt->thr_id = pthread_self();
	// We expect the value of *array to be in args[0].cudaArray in the return
	// packet
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset
	shm_ptr += sizeof(*shmpkt);
	memcpy(shm_ptr, desc, sizeof(*desc));
	shmpkt->len = sizeof(*shmpkt) + sizeof(*desc);
	shmpkt->args[1].arr_argi[0] = width;
	shmpkt->args[1].arr_argi[1] = height;
	shmpkt->args[2].argull = flags;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*array = shmpkt->args[0].cudaArray;
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_MALLOC_PITCH;
	shmpkt->thr_id = pthread_self();
	// We expect the sink to write the value of devPtr to args[0].argull
	// We expect the sink to write the value of pitch to args[1].arr_argi[0]
	shmpkt->args[2].arr_argi[0] = width;
	shmpkt->args[2].arr_argi[1] = height;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();

	*devPtr = (void*)shmpkt->args[0].argull;
	*pitch = shmpkt->args[1].arr_argi[0];
	printd(DBG_DEBUG, "devPtr=%p pitch=%lu\n", *devPtr, *pitch);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpy(void *dst, const void *src,
		size_t count, enum cudaMemcpyKind kind)
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d\n",
			dst, src, count, kind);

	TIMER_DECLARE2(tsetup, twait);
	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyHostToHost:
		{
			shmpkt->method_id = CUDA_MEMCPY_H2H; // why would you call this?
			memcpy(dst, src, count); // right?!
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			// Need to push data DOWN to the gpu
			shmpkt->method_id = CUDA_MEMCPY_H2D;
			shmpkt->args[0].argull = (uintptr_t)dst; // gpu ptr
			shmpkt->args[1].argull = sizeof(*shmpkt);
			shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[1].argull);
			memcpy(shm_ptr, src, count);
			shmpkt->len = sizeof(*shmpkt) + count;
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			// Need to pull data UP from the gpu
			shmpkt->method_id = CUDA_MEMCPY_D2H;
			shmpkt->args[0].argull = sizeof(*shmpkt);
			// We will expect to read 'count' bytes at this ^ offset into dst
			shmpkt->args[1].argull = (uintptr_t)src; // gpu ptr
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_D2D;
			shmpkt->args[0].argull = (uintptr_t)dst; // gpu ptr
			shmpkt->args[1].argull = (uintptr_t)src; // gpu ptr
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		default:
			return cudaErrorInvalidMemcpyDirection;
	}
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->is_sync = true;
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(twait, shmpkt->lat.lib.wait);

	// copy data to user-space region
	if (shmpkt->method_id == CUDA_MEMCPY_D2H) {
		TIMER_RESUME(tsetup);
		shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
		memcpy(dst, shm_ptr, count);
	}
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d stream=%p\n",
			dst, src, count, kind, stream);

	TIMER_DECLARE2(tsetup,twait);
	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyHostToHost:
		{
			shmpkt->method_id = CUDA_MEMCPY_ASYNC_H2H; // why would you call this?
			memcpy(dst, src, count); // right?!
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			// Need to push data DOWN to the gpu
			shmpkt->method_id = CUDA_MEMCPY_ASYNC_H2D;
			shmpkt->args[0].argull = (uintptr_t)dst; // gpu ptr
			shmpkt->args[1].argull = sizeof(*shmpkt);
			shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[1].argull);
			memcpy(shm_ptr, src, count);
			shmpkt->len = sizeof(*shmpkt) + count;
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			// Need to pull data UP from the gpu
			shmpkt->method_id = CUDA_MEMCPY_ASYNC_D2H;
			shmpkt->args[0].argull = sizeof(*shmpkt);
			// We will expect to read 'count' bytes at this ^ offset into dst
			// XXX Since this is an async function, the data won't necessarily
			// be there just yet. For now, we force all async calls to be
			// synchronous until a solution is found.
			shmpkt->args[1].argull = (uintptr_t)src; // gpu ptr
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_D2D;
			shmpkt->args[0].argull = (uintptr_t)dst; // gpu ptr
			shmpkt->args[1].argull = (uintptr_t)src; // gpu ptr
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		default:
			return cudaErrorInvalidMemcpyDirection;
	}
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->args[3].stream = stream;
	shmpkt->is_sync = true;
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(twait, shmpkt->lat.lib.wait);

	// copy data to user-space region
	// XXX We'll only do this now, since these calls are all forced to be
	// synchronous. Once they're actually async, this copy may not be performed
	// at this time.
	if (shmpkt->method_id == CUDA_MEMCPY_D2H) {
		TIMER_RESUME(tsetup);
		shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
		memcpy(dst, shm_ptr, count);
	}
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
	void *shm_ptr;

	printd(DBG_DEBUG, "symb %p\n", symbol);

	TIMER_DECLARE2(tsetup, twait);
	TIMER_START(tsetup);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	shm_ptr = (void*)((uintptr_t)shmpkt + sizeof(*shmpkt));
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyDeviceToHost:
		{
			shmpkt->method_id = CUDA_MEMCPY_FROM_SYMBOL_D2H;
			// offset within the shm from which we'll expect to read the user
			// data in the return packet
			shmpkt->args[0].argull = sizeof(*shmpkt);
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_FROM_SYMBOL_D2D;
			shmpkt->args[0].argull = (uintptr_t)dst; // gpu ptr
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		default:
		{
			fprintf(stderr, USERMSG_PREFIX " memcpy direction %d not implemented\n", kind);
			return cudaErrorInvalidMemcpyDirection;
		}
	}
	if (__func_symb_param_is_string(symbol)) {
		shmpkt->args[1].argull = (shm_ptr - (void*)shmpkt); // store string in shm
		memcpy(shm_ptr, symbol, strlen(symbol) + 1);
		shm_ptr += strlen(symbol) + 1;
		shmpkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", symbol);
		shmpkt->len += strlen(symbol) + 1;
	} else {
		shmpkt->args[1].argull = (uintptr_t)symbol;
	}
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->args[2].arr_argi[1] = offset;
	shmpkt->args[3].argll = kind;
	shmpkt->is_sync = true;
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	shmpkt->flags |= CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(twait, shmpkt->lat.lib.wait);

	if (kind == cudaMemcpyDeviceToHost) {
		TIMER_RESUME(tsetup);
		shm_ptr = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
		memcpy(dst, shm_ptr, count);
	}
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
	void *shm_ptr;

	printd(DBG_DEBUG, "dst=%p wOffset=%lu, hOffset=%lu, src=%p, count=%lu\n",
			dst, wOffset, hOffset, src, count);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].cudaArray = dst; // gpu ptr
	shmpkt->args[1].arr_argi[0] = wOffset;
	shmpkt->args[1].arr_argi[1] = hOffset;
	switch (kind) {
		case cudaMemcpyHostToHost:
		case cudaMemcpyDeviceToHost:
		{
			BUG(0); // wtf is copying d->h or h->h when dst is a device address?
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			// Need to push data DOWN to the gpu
			shmpkt->method_id = CUDA_MEMCPY_TO_ARRAY_H2D;
			shmpkt->args[2].argull = sizeof(*shmpkt);
			shm_ptr = (void*)((uintptr_t)shmpkt + sizeof(*shmpkt));
			memcpy(shm_ptr, src, count);
			shmpkt->len = sizeof(*shmpkt) + count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_TO_ARRAY_D2D;
			shmpkt->args[2].argp = (void*)src; // gpu ptr?
			shmpkt->len = sizeof(*shmpkt);
		}
		break;
		default:
			return cudaErrorInvalidMemcpyDirection;
	}
	shmpkt->args[3].arr_argi[0] = count;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}


cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset __dv(0),
		enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;

	printd(DBG_DEBUG, "symb %p\n", symbol);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	shm_ptr = (void*)((uintptr_t)shmpkt + sizeof(*shmpkt));
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	shmpkt->len = sizeof(*shmpkt);

	if (__func_symb_param_is_string(symbol)) {
		shmpkt->args[0].argull = (shm_ptr - (void*)shmpkt); // offset of string
		memcpy(shm_ptr, symbol, strlen(symbol) + 1);
		shmpkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		shm_ptr += strlen(symbol) + 1;
		printd(DBG_DEBUG, "\tsymb is string: %s\n", symbol);
		shmpkt->len += strlen(symbol) + 1;
	} else {
		shmpkt->args[0].argull = (uintptr_t)symbol;
	}

	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_TO_SYMBOL_H2D;
			shmpkt->args[1].argull = (shm_ptr - (void*)shmpkt);
			memcpy(shm_ptr, src, count);
			shm_ptr += count;
			shmpkt->len += count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_TO_SYMBOL_D2D;
			shmpkt->args[1].argull = (uintptr_t)src;
		}
		break;
		default:
		{
			fprintf(stderr, USERMSG_PREFIX " memcpy direction %d not implemented\n", kind);
			return cudaErrorInvalidMemcpyDirection;
		}
	}
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->args[2].arr_argi[1] = offset;
	shmpkt->args[3].argll = kind;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags |= CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaMemcpyToSymbolAsync(
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0))
{
	struct cuda_packet *shmpkt;
	void *shm_ptr;

	printd(DBG_DEBUG, "symb %p\n", symbol);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	shm_ptr = (void*)((uintptr_t)shmpkt + sizeof(*shmpkt));
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->thr_id = pthread_self();
	shmpkt->len = sizeof(*shmpkt);

	if (__func_symb_param_is_string(symbol)) {
		shmpkt->args[1].argull = (shm_ptr - (void*)shmpkt); // offset of string
		memcpy(shm_ptr, symbol, strlen(symbol) + 1);
		shmpkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		printd(DBG_DEBUG, "\tsymb is string: %s\n", symbol);
		shm_ptr += strlen(symbol) + 1;
		shmpkt->len += strlen(symbol) + 1;
	} else {
		shmpkt->args[1].argull = (uintptr_t)symbol;
	}

	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D;
			shmpkt->args[1].argull = (shm_ptr - (void*)shmpkt);
			memcpy(shm_ptr, src, count);
			shm_ptr += count;
			shmpkt->len += count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			shmpkt->method_id = CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D;
			shmpkt->args[1].argull = (uintptr_t)src;
		}
		break;
		default:
		{
			fprintf(stderr, USERMSG_PREFIX
					" memcpy direction %d not implemented\n", kind);
			return cudaErrorInvalidMemcpyDirection;
		}
	}
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->args[2].arr_argi[1] = offset;
	shmpkt->args[3].stream = stream;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags |= CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_MEMSET;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = (uintptr_t)devPtr;
	shmpkt->args[1].argll = value;
	shmpkt->args[2].arr_argi[0] = count;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	printd(DBG_DEBUG, "called\n");

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_BIND_TEXTURE;
	shmpkt->thr_id = pthread_self();
	// We'll expect the value of *offset to be in args[0].arr_argi[0] within the
	// return packet
	shmpkt->args[0].argp = (void*)texRef; // address
	shmpkt->args[1].texRef = *texRef; // data
	shmpkt->args[2].argp = (void*)devPtr,
	shmpkt->args[3].desc = *desc; // whole struct copy
	shmpkt->args[4].arr_argi[0] = size;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*offset = shmpkt->args[0].arr_argi[0];
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaBindTextureToArray(
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	struct cuda_packet *shmpkt;
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	printd(DBG_DEBUG, "called\n");

	TIMER_DECLARE1(t);
	TIMER_START(t);
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_BIND_TEXTURE_TO_ARRAY;
	shmpkt->thr_id = pthread_self();
	// the caller will customize the values within texRef before invoking this
	// function, thus we need to copy the entire structure as well as its
	// address, so the sink can find the texture it registered with CUDA
	shmpkt->args[0].argp = (void*)texRef; // address
	shmpkt->args[1].texRef = *texRef; // data
	shmpkt->args[2].cudaArray = (struct cudaArray*)array;
	shmpkt->args[3].desc = *desc; // whole struct copy
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
		enum cudaChannelFormatKind format)
{
#if 0
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_DRIVER_GET_VERSION;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt);
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*driverVersion = *((int*)((uintptr_t)shmpkt + shmpkt->args[0].argull));
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	struct cuda_packet *shmpkt;

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_RUNTIME_GET_VERSION;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt);
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	*runtimeVersion = *((int*)((uintptr_t)shmpkt + shmpkt->args[0].argull));
	update_latencies(&shmpkt->lat);
	return shmpkt->ret_ex_val.err;
}

//
// Undocumented API
//

void** __cudaRegisterFatBinary(void* cubin)
{
	int err, cubin_size;
	struct cuda_packet *shmpkt;
	void *cubin_shm; // pointer to serialized copy of argument
	cache_num_entries_t entries_in_cubin;

	num_registered_cubins++;

	err = attach_assembly_runtime(); // will return if already done
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		assert(0);
	}

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset((void*)shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = __CUDA_REGISTER_FAT_BINARY;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset

	// Serialize the complex cubin structure into the shared memory region,
	// immediately after the location of the cuda packet.
	cubin_shm = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memset(&entries_in_cubin, 0, sizeof(entries_in_cubin));
	cubin_size = getFatRecPktSize(cubin, &entries_in_cubin);
	printd(DBG_DEBUG, "size of cubin: %d bytes\n", cubin_size);
	if (cubin_size >= THREAD_SHM_SIZE) {
		fprintf(stderr, "%s: error: cubin size too large: %d\n",
				__func__, cubin_size);
		assert(0);
	}
	err = packFatBinary((char *)cubin_shm, cubin, &entries_in_cubin);
	if (err < 0) {
		printd(DBG_ERROR, "error calling packFatBinary\n");
		return NULL;
	}
	shmpkt->args[1].argll = cubin_size;
	shmpkt->len = sizeof(*shmpkt) + cubin_size;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb(); // flush writes from caches
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	printd(DBG_DEBUG, "handle=%p\n", shmpkt->ret_ex_val.handle);
	update_latencies(&shmpkt->lat);
	return (void**)(shmpkt->ret_ex_val.handle);
}

void __cudaUnregisterFatBinary(void** fatCubinHandle)
{
	struct cuda_packet *shmpkt;
	printd(DBG_INFO, "handle=%p\n", fatCubinHandle);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	num_registered_cubins--;
	shmpkt->method_id = __CUDA_UNREGISTER_FAT_BINARY;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argdp = fatCubinHandle;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	int err;
	struct cuda_packet *shmpkt;
	void *var = NULL;

	printd(DBG_DEBUG, "handle=%p hostFun=%p deviceFun=%s deviceName=%s\n",
			fatCubinHandle, hostFun, deviceFun, deviceName);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = __CUDA_REGISTER_FUNCTION;
	shmpkt->thr_id = pthread_self();

	shmpkt->args[0].argull = sizeof(*shmpkt);
	// now pack it into the shm
	var = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	err = packRegFuncArgs(var, fatCubinHandle, hostFun, deviceFun,
			deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
	if (err < 0) {
		printd(DBG_ERROR, "Error packing arguments\n");
		assert(0); // FIXME Is there a better strategy to failing?
	}
	shmpkt->args[1].arr_argi[0] =
		getSize_regFuncArgs(fatCubinHandle, hostFun, deviceFun, deviceName,
				thread_limit, tid, bid, bDim, gDim, wSize);
	shmpkt->len = sizeof(*shmpkt) + shmpkt->args[1].arr_argi[0];
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	int err;
	struct cuda_packet *shmpkt;
	void *var = NULL;

	printd(DBG_DEBUG, "symbol=%p\n", hostVar);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = __CUDA_REGISTER_VARIABLE;
	shmpkt->thr_id = pthread_self();

	shmpkt->args[0].argull = sizeof(*shmpkt);
	// now pack it into the shm
	var = (void*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	err = packRegVar(var, fatCubinHandle, hostVar, deviceAddress, deviceName,
			ext, vsize, constant, global);
	if (err < 0) {
		printd(DBG_ERROR, "Error packing arguments\n");
		assert(0); // FIXME Is there a better strategy to failing?
	}
	shmpkt->args[1].arr_argi[0]
		= getSize_regVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
				ext, vsize, constant, global);
	shmpkt->len = sizeof(*shmpkt) + shmpkt->args[1].arr_argi[0];
	shmpkt->is_sync = true;

	// Add it to our list of known variable symbols.
	registered_vars[num_registered_vars++] = (uintptr_t)hostVar;
	if (num_registered_vars >= MAX_REGISTERED_VARS) {
		fprintf(stderr, USERMSG_PREFIX " exceeded allowance on num vars to register\n");
		BUG(1);
	}
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
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
	shmpkt->flags = CUDA_PKT_REQUEST;
	wmb();
	while (!(shmpkt->flags & CUDA_PKT_RESPONSE))
		rmb();
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return;
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

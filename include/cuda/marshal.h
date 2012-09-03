/**
 * @file marshal.h
 *
 * @date Mar 1, 2011
 * @author Magda Slawinska, magg@gatech.edu
 * 	- original author
 *
 * @date 2011-12-18
 * @author Alex Merritt, merritt.alex@gatech.edu
 * - Cleaned out obsolete functions
 * 2012-04-07
 * - Added all the public calls as inline functions. All public functions
 *   marshal into a cuda_packet as opposed to a random buffer.
 */

#ifndef MARSHAL_H_
#define MARSHAL_H_

// CUDA includes
#include <__cudaFatFormat.h>

// Project includes
#include <cuda/fatcubininfo.h>
#include <cuda/packet.h>

/*-------------------------------------- INTERNAL STATE ----------------------*/

#define MAX_REGISTERED_VARS 5210
//! Symbol addresses from __cudaRegisterVar. Used to determine if the symbol
//! parameter in certain functions is actually the address of a variable, or
//! the string name of one of the variables in functions which accept symbols.
//! TODO Make this cleaner code.
static uintptr_t registered_vars[MAX_REGISTERED_VARS];
static unsigned int num_registered_vars = 0;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

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

/*-------------------------------------- HIDDEN CALLS ------------------------*/

#define OK 0
#define ERROR -1

/**
 * For storing the number of records  for particular structures
 * contained in the __cudaFatCubinBinaryRec
 */
typedef struct {
	int nptxs;
	int ncubs;
	int ndebs;
	int ndeps; // number of dependends
	int nelves;
	int nexps; // number of exported
	int nimps; // number of imported
} cache_num_entries_t;

/**
 * To use it when marshaling and unmarshaling in the sent packet
 * Should indicate the size of the following bytes
 */
typedef unsigned int size_pkt_field_t;

int mallocCheck(const void * const p, const char * const pFuncName,
		const char * pExtraMsg);

int getFatRecPktSize(const __cudaFatCudaBinary *pFatCubin,
		cache_num_entries_t * pEntriesCache);
//int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num);

int packFatBinary(char * pFatPack, __cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache);
int unpackFatBinary(__cudaFatCudaBinary *pFatC, char * pFatPack);

int getSize_regFuncArgs(void** fatCubinHandle, const char* hostFun,
        char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
        uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
int packRegFuncArgs(void *dst, void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
int unpackRegFuncArgs(reg_func_args_t * pRegFuncArgs, char * pPacket);

int getSize_regVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
		const char *deviceName, int ext, int vsize,int constant, int global);
int packRegVar(void *dst, void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize, int
		constant, int global);
int unpackRegVar(reg_var_args_t * pRegVar, char *pPacket);

int freeRegFunc(reg_func_args_t *args);
int freeFatBinary(__cudaFatCudaBinary *fatCubin);
int freeRegVar(reg_var_args_t *args);

/**
 * cleans the structure, frees the allocated memory, sets values to zeros,
 * nulls, etc; intended to be used in __unregisterCudaFatBinary
 */
int cleanFatCubinInfo(struct cuda_fatcubin_info * pFatCInfo);

/*-------------------------------------- DEFINITIONS FOR PUBLIC CALLS --------*/

/* Directly write the cuda packet into the shared memory region. Indicate where
 * the output argument's data should be stored.
 *
 *   shm	Offset
 * +------+ 0
 * + cuda +
 * + pkt  +
 * +------+	sizeof(pkt)
 * + data + 
 * +------+ sizeof(pkt) + (data length)
 *
 * pkt->len = sizeof(pkt) + (data length)
 *
 * Give the packet one argument indicating the offset into the shared memory
 * region where it expects the value of 'count' to reside. This will be copied
 * into the user-provided address before returning. For functions with
 * additional pointer arguments, append them to the bottom and do additional
 * copying before returning.
 *
 * The functions below don't assume the cuda_packet is stored in the memory
 * buffer, as they accept a separate argument for the marshaled argument data.
 */

//
// Thread Management API
//

static inline void
pack_cudaThreadExit(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_THREAD_EXIT;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
pack_cudaThreadSynchronize(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_THREAD_SYNCHRONIZE;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

//
// Device Management API
//

// Assembly code executes cudaGetDevice; must match this function.
static inline void
pack_cudaGetDevice(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_GET_DEVICE;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
	// Expect the device ID in args[0].argll
}

static inline void
insert_cudaGetDevice(struct cuda_packet *pkt, int device)
{
	pkt->args[0].argll = device;
}

static inline void
extract_cudaGetDevice(struct cuda_packet *pkt, int *device)
{
	*device = pkt->args[0].argll;
}

static inline void
pack_cudaGetDeviceCount(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_GET_DEVICE_COUNT;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
	// Expect the count in args[0].argll
}

static inline void
insert_cudaGetDeviceCount(struct cuda_packet *pkt, int count)
{
	pkt->args[0].argll = count;
}

static inline void
extract_cudaGetDeviceCount(struct cuda_packet *pkt, int *count)
{
	*count = pkt->args[0].argll;
}

static inline void
pack_cudaGetDeviceProperties(struct cuda_packet *pkt, int device)
{
	pkt->method_id = CUDA_GET_DEVICE_PROPERTIES;
	pkt->thr_id = pthread_self();
	pkt->args[0].argull = 0UL; // Expect structure at this offset in buffer
	pkt->args[1].argll = device;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaGetDeviceProperties(struct cuda_packet *pkt,
		int *device)
{
	*device = pkt->args[1].argll;
}

static inline void
insert_cudaGetDeviceProperties(struct cuda_packet *pkt, void *buf,
		struct cudaDeviceProp *prop)
{
	void *loc = (void*)((uintptr_t)buf + pkt->args[0].argull);
	memcpy(loc, prop, sizeof(*prop));
}

static inline void
extract_cudaGetDeviceProperties(struct cuda_packet *pkt, void *buf,
		struct cudaDeviceProp *prop)
{
	void *loc = (void*)((uintptr_t)buf + pkt->args[0].argull);
	memcpy(prop, loc, sizeof(*prop));
}

static inline void
pack_cudaSetDevice(struct cuda_packet *pkt,
		int device)
{
	pkt->method_id = CUDA_SET_DEVICE;
	pkt->thr_id = pthread_self();
	pkt->args[0].argll = device;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaSetDevice(struct cuda_packet *pkt,
		int *device)
{
	*device = pkt->args[0].argll;
}

static inline void
pack_cudaSetDeviceFlags(struct cuda_packet *pkt,
		unsigned int flags)
{
	pkt->method_id = CUDA_SET_DEVICE_FLAGS;
	pkt->thr_id = pthread_self();
	pkt->args[0].argull = flags;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaSetDeviceFlags(struct cuda_packet *pkt,
		unsigned int *flags)
{
	*flags = (unsigned int)pkt->args[0].argull;
}

static inline void
pack_cudaSetValidDevices(struct cuda_packet *pkt, void *buf,
		int *device_arr, int len) // len is not num bytes
{
	size_t arr_len = (len * sizeof(*device_arr));
	pkt->method_id = CUDA_SET_VALID_DEVICES;
	pkt->thr_id = pthread_self();
	pkt->args[0].argull = 0UL; // device_arr is at this offset in buf
	pkt->args[1].argll = len;
	memcpy(buf, device_arr, arr_len);
	pkt->len = sizeof(*pkt) + arr_len;
	pkt->is_sync = method_synctable[pkt->method_id];
}

// does not copy the array out of buf
static inline void
unpack_cudaSetValidDevices(struct cuda_packet *pkt, void *buf,
		int **device_arr, int *len)
{
	*device_arr = (int*)((uintptr_t)buf + pkt->args[0].argull);
	*len = pkt->args[1].argll;
}

//
// Stream Management API
//

static inline void
pack_cudaStreamCreate(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_STREAM_CREATE;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
insert_cudaStreamCreate(struct cuda_packet *pkt,
		cudaStream_t stream)
{
	pkt->args[0].stream = stream;
}

static inline void
extract_cudaStreamCreate(struct cuda_packet *pkt,
		cudaStream_t *pStream)
{
	*pStream = pkt->args[0].stream;
}

static inline void
pack_cudaStreamDestroy(struct cuda_packet *pkt,
		cudaStream_t stream)
{
	pkt->method_id = CUDA_STREAM_DESTROY;
	pkt->thr_id = pthread_self();
	pkt->args[0].stream = stream;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaStreamDestroy(struct cuda_packet *pkt,
		cudaStream_t *pStream)
{
	*pStream = pkt->args[0].stream;
}

static inline void
pack_cudaStreamQuery(struct cuda_packet *pkt,
		cudaStream_t stream)
{
	pkt->method_id = CUDA_STREAM_DESTROY;
	pkt->thr_id = pthread_self();
	pkt->args[0].stream = stream;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaStreamQuery(struct cuda_packet *pkt,
		cudaStream_t *pStream)
{
	*pStream = pkt->args[0].stream;
}

static inline void
pack_cudaStreamSynchronize(struct cuda_packet *pkt,
		cudaStream_t stream)
{
	pkt->method_id = CUDA_STREAM_SYNCHRONIZE;
	pkt->thr_id = pthread_self();
	pkt->args[0].stream = stream;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaStreamSynchronize(struct cuda_packet *pkt,
		cudaStream_t *pStream)
{
	*pStream = pkt->args[0].stream;
}

//
// Execution Control API
//

static inline void
pack_cudaConfigureCall(struct cuda_packet *pkt,
		dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
	pkt->method_id = CUDA_CONFIGURE_CALL;
	pkt->thr_id = pthread_self();
	pkt->args[0].arg_dim = gridDim; // = on structs works :)
	pkt->args[1].arg_dim = blockDim;
	pkt->args[2].arr_argi[0] = sharedMem;
	pkt->args[3].stream = stream;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaConfigureCall(struct cuda_packet *pkt,
		dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
		cudaStream_t *stream)
{
	*gridDim = pkt->args[0].arg_dim;
	*blockDim = pkt->args[1].arg_dim;
	*sharedMem = pkt->args[2].arr_argi[0];
	*stream = pkt->args[3].stream;
}

// TODO
// 		cudaFuncGetAttributes

static inline void
pack_cudaLaunch(struct cuda_packet *pkt,
		const char *entry)
{
	pkt->method_id = CUDA_LAUNCH;
	pkt->thr_id = pthread_self();
	// FIXME We assume entry is just a memory pointer, not a string.
	pkt->args[0].argull = (uintptr_t)entry;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaLaunch(struct cuda_packet *pkt,
		const char **entry)
{
	*entry = (const char*)pkt->args[0].argull;
}

static inline void
pack_cudaSetupArgument(struct cuda_packet *pkt, void *buf,
		const void *arg, size_t size, size_t offset)
{
	pkt->method_id = CUDA_SETUP_ARGUMENT;
	pkt->thr_id = pthread_self();
	pkt->args[0].argull = 0UL; // offset of arg within buf
	memcpy(buf, arg, size);
	pkt->args[1].arr_argi[0] = size;
	pkt->args[1].arr_argi[1] = offset;
	pkt->len = sizeof(*pkt) + size;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaSetupArgument(struct cuda_packet *pkt, void *buf,
		const void **arg, size_t *size, size_t *offset)
{
	*arg = (void*)((uintptr_t)buf + pkt->args[0].argull);
	*size = pkt->args[1].arr_argi[0];
	*offset = pkt->args[1].arr_argi[1];
}

//
// Memory Management API
//

static inline void
pack_cudaFree(struct cuda_packet *pkt, void *devPtr)
{
	pkt->method_id = CUDA_FREE;
	pkt->thr_id = pthread_self();
	pkt->args[0].argp = devPtr;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaFree(struct cuda_packet *pkt,
		void **devPtr)
{
	*devPtr = pkt->args[0].argp;
}

static inline void
pack_cudaFreeArray(struct cuda_packet *pkt,
		struct cudaArray *array)
{
	pkt->method_id = CUDA_FREE_ARRAY;
	pkt->thr_id = pthread_self();
	pkt->args[0].cudaArray = array;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaFreeArray(struct cuda_packet *pkt,
		struct cudaArray **array)
{
	*array = pkt->args[0].cudaArray;
}

// TODO
// 		cudaFreeHost
// 		cudaHostAlloc

static inline void
pack_cudaMalloc(struct cuda_packet *pkt, size_t size)
{
	pkt->method_id = CUDA_MALLOC;
	pkt->thr_id = pthread_self();
	pkt->args[1].arr_argi[0] = size;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMalloc(struct cuda_packet *pkt, size_t *size)
{
	*size = pkt->args[1].arr_argi[0];
}

static inline void
insert_cudaMalloc(struct cuda_packet *pkt, void *devPtr)
{
	pkt->args[0].argp = devPtr;
}

static inline void
extract_cudaMalloc(struct cuda_packet *pkt, void **devPtr)
{
	*devPtr = pkt->args[0].argp;
}

static inline void
pack_cudaMallocArray(struct cuda_packet *pkt,
		const struct cudaChannelFormatDesc *desc, size_t width, size_t height,
		unsigned int flags)
{
	pkt->method_id = CUDA_MALLOC_ARRAY;
	pkt->thr_id = pthread_self();
	pkt->args[0].desc = *desc;
	pkt->len = sizeof(*pkt);
	pkt->args[1].arr_argi[0] = width;
	pkt->args[1].arr_argi[1] = height;
	pkt->args[2].argull = flags;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMallocArray(struct cuda_packet *pkt,
		struct cudaChannelFormatDesc **desc, size_t *width, size_t *height,
		unsigned int *flags)
{
	*desc = &pkt->args[0].desc;
	*width = pkt->args[1].arr_argi[0];
	*height = pkt->args[1].arr_argi[1];
	*flags = pkt->args[2].argull;
}

static inline void
insert_cudaMallocArray(struct cuda_packet *pkt, struct cudaArray *array)
{
	pkt->args[3].cudaArray = array;
}

static inline void
extract_cudaMallocArray(struct cuda_packet *pkt, struct cudaArray **array)
{
	*array = pkt->args[3].cudaArray; // just a device address
}

static inline void
pack_cudaMallocPitch(struct cuda_packet *pkt,
		size_t width, size_t height)
{
	pkt->method_id = CUDA_MALLOC_PITCH;
	pkt->thr_id = pthread_self();
	pkt->args[2].arr_argi[0] = width;
	pkt->args[2].arr_argi[1] = height;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMallocPitch(struct cuda_packet *pkt,
		size_t *width, size_t *height)
{
	*width = pkt->args[2].arr_argi[0];
	*height = pkt->args[2].arr_argi[1];
}

static inline void
insert_cudaMallocPitch(struct cuda_packet *pkt,
		void *devPtr, size_t pitch)
{
	pkt->args[0].argp = devPtr;
	pkt->args[1].arr_argi[0] = pitch;
}

static inline void
extract_cudaMallocPitch(struct cuda_packet *pkt,
		void **devPtr, size_t *pitch)
{
	*devPtr = pkt->args[0].argp;
	*pitch = pkt->args[1].arr_argi[0];
}

static inline void
pack_cudaMemcpy(struct cuda_packet *pkt, void *buf,
		void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	pkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyHostToHost:
		{
			pkt->method_id = CUDA_MEMCPY_H2H; // why would you call this?
			memcpy(dst, src, count); // right?!
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			// Need to push data DOWN to the gpu
			pkt->method_id = CUDA_MEMCPY_H2D;
			pkt->args[0].argp = dst; // gpu ptr
			pkt->args[1].argull = 0UL; // offset of src in buf
			memcpy(buf, src, count);
			pkt->len = sizeof(*pkt) + count;
			pkt->is_sync = method_synctable[pkt->method_id];
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			// Need to pull data UP from the gpu
			pkt->method_id = CUDA_MEMCPY_D2H;
			// Copy 'count' bytes at this offset into dst later
			pkt->args[0].argull = 0UL;
			pkt->args[1].argp = (void*)src; // gpu ptr
			pkt->len = sizeof(*pkt);
			pkt->is_sync = method_synctable[pkt->method_id];
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_D2D;
			pkt->args[0].argp = dst; // gpu ptr
			pkt->args[1].argp = (void*)src; // gpu ptr
			pkt->len = sizeof(*pkt);
			pkt->is_sync = method_synctable[pkt->method_id];
		}
		break;
		default:
			BUG(1);
	}
	pkt->args[2].arr_argi[0] = count;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMemcpy(struct cuda_packet *pkt, void *buf,
		void **dst, const void **src, size_t *count, enum cudaMemcpyKind kind)
{
	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			*dst = pkt->args[0].argp; // gpu ptr
			*src = (void*)((uintptr_t)buf + pkt->args[1].argull);
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			*dst = (void*)((uintptr_t)buf + pkt->args[0].argull);
			*src = pkt->args[1].argp; // gpu ptr
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			*dst = pkt->args[0].argp; // gpu ptr
			*src = pkt->args[1].argp; // gpu ptr
		}
		break;
		default:
			BUG(1);
	}
	*count = pkt->args[2].arr_argi[0];
}

static inline void
extract_cudaMemcpy(struct cuda_packet *pkt, void *buf,
		void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	if (pkt->method_id == CUDA_MEMCPY_D2H)
		memcpy(dst, (void*)((uintptr_t)buf + pkt->args[0].argull), count);
}

static inline void
pack_cudaMemcpyAsync(struct cuda_packet *pkt, void *buf,
		void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
		cudaStream_t stream)
{
	// Very similar (almost same) as non-async version
	pkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyHostToHost:
		{
			pkt->method_id = CUDA_MEMCPY_ASYNC_H2H;
			memcpy(dst, src, count);
		}
		break;
		case cudaMemcpyHostToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_ASYNC_H2D;
			pkt->args[0].argull = (uintptr_t)dst;
			pkt->args[1].argull = 0UL;
			memcpy(buf, src, count);
			pkt->len = sizeof(*pkt) + count;
		}
		break;
		case cudaMemcpyDeviceToHost:
		{
			pkt->method_id = CUDA_MEMCPY_ASYNC_D2H;
			pkt->args[0].argull = 0UL;
			pkt->args[1].argull = (uintptr_t)src;
			pkt->len = sizeof(*pkt);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_D2D;
			pkt->args[0].argull = (uintptr_t)dst;
			pkt->args[1].argull = (uintptr_t)src;
			pkt->len = sizeof(*pkt);
		}
		break;
		default:
			BUG(1);
	}
	pkt->args[2].arr_argi[0] = count;
	pkt->args[3].stream = stream;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
extract_cudaMemcpyAsync(struct cuda_packet *pkt, void *buf,
		void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
		cudaStream_t stream)
{
	if (pkt->method_id == CUDA_MEMCPY_D2H)
		memcpy(dst, (void*)((uintptr_t)buf + pkt->args[0].argull), count);
}

static inline void
pack_cudaMemcpyFromSymbol(struct cuda_packet *pkt, void *buf,
		void *dst, const char *symbol, size_t count, size_t offset,
		enum cudaMemcpyKind kind)
{
	pkt->thr_id = pthread_self();
	switch (kind) {
		case cudaMemcpyDeviceToHost:
		{
			pkt->method_id = CUDA_MEMCPY_FROM_SYMBOL_D2H;
			pkt->args[0].argull = 0UL; // offset we'll read from
			pkt->len = sizeof(*pkt);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_FROM_SYMBOL_D2D;
			pkt->args[0].argp = dst; // gpu ptr
			pkt->len = sizeof(*pkt);
		}
		break;
		default:
			BUG(1);
	}
	if (__func_symb_param_is_string(symbol)) {
		pkt->args[1].argull = 0UL; // offset into buf we put the str
		memcpy(buf, symbol, strlen(symbol) + 1);
		pkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", symbol);
		pkt->len += strlen(symbol) + 1;
	} else {
		pkt->args[1].argp = (void*)symbol;
	}
	pkt->args[2].arr_argi[0] = count;
	pkt->args[2].arr_argi[1] = offset;
	pkt->args[3].argll = kind;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMemcpyFromSymbol(struct cuda_packet *pkt, void *buf,
		void **dst, const char **symbol, size_t *count, size_t *offset,
		enum cudaMemcpyKind kind)
{
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		*symbol = (const char*)((uintptr_t)buf + pkt->args[1].argull);
	} else {
		*symbol = (const char*)(pkt->args[1].argp);
	}
	switch (kind) {
		case cudaMemcpyDeviceToHost:
		{
			*dst = (void*)((uintptr_t)buf + pkt->args[0].argull);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			*dst = pkt->args[0].argp;
		}
		break;
		default:
			BUG(1);
	}
	*count = pkt->args[2].arr_argi[0];
	*offset = pkt->args[2].arr_argi[1];
	printd(DBG_DEBUG, "memcpyFromSymb symb=%p count=%lu\n", *symbol, *count);
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", *symbol);
	}
}

static inline void
extract_cudaMemcpyFromSymbol(struct cuda_packet *pkt, void *buf,
		void *dst, const char *symbol, size_t count, size_t offset,
		enum cudaMemcpyKind kind)
{
	if (kind == cudaMemcpyDeviceToHost)
		memcpy(dst, (void*)((uintptr_t)buf + pkt->args[0].argull), count);
}

static inline void
pack_cudaMemcpyToArray(struct cuda_packet *pkt, void *buf,
		struct cudaArray *dst, size_t wOffset, size_t hOffset,
		const void *src, size_t count, enum cudaMemcpyKind kind)
{
	pkt->thr_id = pthread_self();
	pkt->args[0].cudaArray = dst; // gpu ptr
	pkt->args[1].arr_argi[0] = wOffset;
	pkt->args[1].arr_argi[1] = hOffset;
	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			// Need to push data DOWN to the gpu
			pkt->method_id = CUDA_MEMCPY_TO_ARRAY_H2D;
			pkt->args[2].argull = 0UL; // offset in buf src is located
			memcpy(buf, src, count);
			pkt->len = sizeof(*pkt) + count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_TO_ARRAY_D2D;
			pkt->args[2].argp = (void*)src; // gpu ptr?
			pkt->len = sizeof(*pkt);
		}
		break;
		default:
			BUG(1);
	}
	pkt->args[3].arr_argi[0] = count;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMemcpyToArray(struct cuda_packet *pkt, void *buf,
		struct cudaArray **dst, size_t *wOffset, size_t *hOffset,
		const void **src, size_t *count, enum cudaMemcpyKind kind)
{
	*dst = pkt->args[0].cudaArray; // gpu ptr
	*wOffset = pkt->args[1].arr_argi[0];
	*hOffset = pkt->args[1].arr_argi[1];
	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			*src = (void*)((uintptr_t)buf + pkt->args[2].argull);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			*src = pkt->args[2].argp;
		}
		break;
		default:
			BUG(1);
	}
	*count = pkt->args[3].arr_argi[0];
}

static inline void
pack_cudaMemcpyToSymbol(struct cuda_packet *pkt, void *buf,
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind)
{
	size_t buf_offset = 0UL;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);

	if (__func_symb_param_is_string(symbol)) {
		pkt->args[0].argull = buf_offset; // for symbol string
		memcpy((void*)((uintptr_t)buf + buf_offset), symbol, strlen(symbol) + 1);
		pkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		buf_offset += strlen(symbol) + 1;
		printd(DBG_DEBUG, "\tsymb is string: %s\n", symbol);
		pkt->len += strlen(symbol) + 1;
	} else {
		pkt->args[0].argp = (void*)symbol;
	}

	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_TO_SYMBOL_H2D;
			pkt->args[1].argull = buf_offset;
			memcpy((void*)((uintptr_t)buf + buf_offset), src, count);
			buf_offset += count;
			pkt->len += count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_TO_SYMBOL_D2D;
			pkt->args[1].argp = (void*)src;
		}
		break;
		default:
			BUG(1);
	}
	pkt->args[2].arr_argi[0] = count;
	pkt->args[2].arr_argi[1] = offset;
	pkt->args[3].argll = kind;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMemcpyToSymbol(struct cuda_packet *pkt, void *buf,
		const char **symbol, const void **src, size_t *count,
		size_t *offset, enum cudaMemcpyKind kind)
{
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		*symbol = (void*)((uintptr_t)buf + pkt->args[0].argull);
	} else {
		*symbol = pkt->args[0].argp;
	}
	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			*src = (void*)((uintptr_t)buf + pkt->args[1].argull);
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			*src = pkt->args[1].argp;
		}
		break;
		default:
			BUG(1);
	}
	*count = pkt->args[2].arr_argi[0];
	*offset = pkt->args[2].arr_argi[1];
	printd(DBG_DEBUG, "memcpyToSymb symb=%p count=%lu\n", *symbol, *count);
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", *symbol);
	}
}

static inline void
pack_cudaMemcpyToSymbolAsync(struct cuda_packet *pkt, void *buf,
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	size_t buf_offset = 0UL; // into buf
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);

	if (__func_symb_param_is_string(symbol)) {
		pkt->args[0].argull = buf_offset; // of symbol string
		memcpy((void*)((uintptr_t)buf + buf_offset), symbol, strlen(symbol) + 1);
		pkt->flags |= CUDA_PKT_SYMB_IS_STRING;
		printd(DBG_DEBUG, "\tsymb is string: %s\n", symbol);
		buf_offset += strlen(symbol) + 1;
		pkt->len += strlen(symbol) + 1;
	} else {
		pkt->args[0].argull = (uintptr_t)symbol;
	}

	switch (kind) {
		case cudaMemcpyHostToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D;
			pkt->args[1].argull = buf_offset;
			memcpy((void*)((uintptr_t)buf + buf_offset), src, count);
			buf_offset += count;
			pkt->len += count;
		}
		break;
		case cudaMemcpyDeviceToDevice:
		{
			pkt->method_id = CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D;
			pkt->args[1].argull = (uintptr_t)src;
		}
		break;
		default:
			BUG(1);
	}
	pkt->args[2].arr_argi[0] = count;
	pkt->args[2].arr_argi[1] = offset;
	pkt->args[3].stream = stream;
	pkt->is_sync = method_synctable[pkt->method_id];
}

// TODO
// 		cudaMemGetInfo

static inline void
pack_cudaMemGetInfo(struct cuda_packet *pkt)
{
    pkt->method_id = CUDA_MEM_GET_INFO;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
}

static inline void
insert_cudaMemGetInfo(struct cuda_packet *pkt,
        size_t free, size_t total)
{
	pkt->args[0].arr_argi[0] = free;
	pkt->args[0].arr_argi[1] = total;
}

static inline void
extract_cudaMemGetInfo(struct cuda_packet *pkt,
        size_t *free, size_t *total)
{
	*free = pkt->args[0].arr_argi[0];
	*total = pkt->args[0].arr_argi[1];
}

static inline void
pack_cudaMemset(struct cuda_packet *pkt,
		void *devPtr, int value, size_t count)
{
	pkt->method_id = CUDA_MEMSET;
	pkt->thr_id = pthread_self();
	pkt->args[0].argp = devPtr;
	pkt->args[1].argll = value;
	pkt->args[2].arr_argi[0] = count;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaMemset(struct cuda_packet *pkt,
		void **devPtr, int *value, size_t *count)
{
	*devPtr = pkt->args[0].argp;
	*value = pkt->args[1].argll;
	*count = pkt->args[2].arr_argi[0];
}

//
// Texture Management API
//

static inline void
pack_cudaBindTexture(struct cuda_packet *pkt,
		const struct textureReference *texRef, const void *devPtr,
		const struct cudaChannelFormatDesc *desc, size_t size)
{
	pkt->method_id = CUDA_BIND_TEXTURE;
	pkt->thr_id = pthread_self();
	pkt->args[0].argp = (void*)texRef; // address
	pkt->args[1].texRef = *texRef; // data
	pkt->args[2].argp = (void*)devPtr,
	pkt->args[3].desc = *desc; // whole struct copy
	pkt->args[4].arr_argi[0] = size;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaBindTexture(struct cuda_packet *pkt,
		const struct textureReference **texRef,
		const struct textureReference **texRef_data,
		const void **devPtr, const struct cudaChannelFormatDesc **desc,
		size_t *size)
{
	*texRef = pkt->args[0].argp; // address for lookup
	*texRef_data = &pkt->args[1].texRef; // data (app may have changed it)
	*devPtr = pkt->args[2].argp;
	*desc = &pkt->args[3].desc;
	*size = pkt->args[4].arr_argi[0];
}

static inline void
insert_cudaBindTexture(struct cuda_packet *pkt,
		size_t offset)
{
	pkt->args[0].arr_argi[0] = offset;
}

static inline void
extract_cudaBindTexture(struct cuda_packet *pkt,
		size_t *offset)
{
	*offset = pkt->args[0].arr_argi[0];
}

static inline void
pack_cudaBindTextureToArray(struct cuda_packet *pkt,
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	pkt->method_id = CUDA_BIND_TEXTURE_TO_ARRAY;
	pkt->thr_id = pthread_self();
	// the caller (application) will customize the values within texRef before
	// invoking this function, thus we need to copy the entire structure as well
	// as its address, so the sink can find the texture it registered with CUDA
	pkt->args[0].argp = (void*)texRef; // address
	pkt->args[1].texRef = *texRef; // data
	pkt->args[2].cudaArray = (struct cudaArray*)array;
	pkt->args[3].desc = *desc; // whole struct copy
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaBindTextureToArray(struct cuda_packet *pkt,
		const struct textureReference **texRef,
		const struct textureReference **texRef_data,
		const struct cudaArray **array,
		const struct cudaChannelFormatDesc **desc)
{
	*texRef = pkt->args[0].argp;
	*texRef_data = &pkt->args[1].texRef;
	*array = pkt->args[2].cudaArray;
	*desc = &pkt->args[3].desc;
}

// TODO
// 		cudaCreateChannelDesc

//
// Version Management API
//

static inline void
pack_cudaDriverGetVersion(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_DRIVER_GET_VERSION;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
insert_cudaDriverGetVersion(struct cuda_packet *pkt,
		int driverVersion)
{
	pkt->args[0].argll = driverVersion;
}

static inline void
extract_cudaDriverGetVersion(struct cuda_packet *pkt,
		int *driverVersion)
{
	*driverVersion = pkt->args[0].argll;
}

static inline void
pack_cudaRuntimeGetVersion(struct cuda_packet *pkt)
{
	pkt->method_id = CUDA_RUNTIME_GET_VERSION;
	pkt->thr_id = pthread_self();
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
	// Expect version in args[0].argll
}

static inline void
insert_cudaRuntimeGetVersion(struct cuda_packet *pkt,
		int runtimeVersion)
{
	pkt->args[0].argll = runtimeVersion;
}

static inline void
extract_cudaRuntimeGetVersion(struct cuda_packet *pkt,
		int *runtimeVersion)
{
	*runtimeVersion = pkt->args[0].argll;
}

//
// Undocumented API
//


static inline void
pack_cudaRegisterFatBinary(struct cuda_packet *pkt, void *buf,
		void *cubin)
{
	int cubin_size, err;
	cache_num_entries_t entries_in_cubin;
	pkt->method_id = __CUDA_REGISTER_FAT_BINARY;
	pkt->thr_id = pthread_self();
	pkt->args[0].argull = 0UL; // offset into buf where cubin will be

	// Serialize the complex cubin structure into the buffer
	memset(&entries_in_cubin, 0, sizeof(entries_in_cubin));
	cubin_size = getFatRecPktSize(cubin, &entries_in_cubin);
	printd(DBG_DEBUG, "size of cubin: %d bytes\n", cubin_size);
	err = packFatBinary((char*)buf, cubin, &entries_in_cubin);
	if (err < 0) BUG(1);
	pkt->args[1].argll = cubin_size;
	pkt->len = sizeof(*pkt) + cubin_size;
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaRegisterFatBinary(struct cuda_packet *pkt, void *buf,
		__cudaFatCudaBinary *cubin)
{
	if (0 > unpackFatBinary(cubin, (char*)((uintptr_t)buf + pkt->args[0].argull)))
		BUG(1);
}

static inline void
extract_cudaRegisterFatBinary(struct cuda_packet *pkt, void ***handle)
{
	*handle = pkt->ret_ex_val.handle;
}

static inline void
pack_cudaUnregisterFatBinary(struct cuda_packet *pkt,
		void **fatCubinHandle)
{
	pkt->method_id = __CUDA_UNREGISTER_FAT_BINARY;
	pkt->thr_id = pthread_self();
	pkt->args[0].argdp = fatCubinHandle;
	pkt->len = sizeof(*pkt);
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaUnregisterFatBinary(struct cuda_packet *pkt,
		void ***fatCubinHandle)
{
	*fatCubinHandle = pkt->args[0].argdp;
}

static inline void
pack_cudaRegisterFunction(struct cuda_packet *pkt, void *buf,
		void** fatCubinHandle, const char* hostFun, char* deviceFun,
		const char* deviceName, int thread_limit, uint3* tid, uint3* bid,
		dim3* bDim, dim3* gDim, int* wSize)
{
	int err;
	pkt->method_id = __CUDA_REGISTER_FUNCTION;
	pkt->thr_id = pthread_self();

	// now pack it into the shm
	pkt->args[0].argull = 0UL; // offset into buf where args will be
	err = packRegFuncArgs(buf, fatCubinHandle, hostFun, deviceFun,
			deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
	if (err < 0) {
		printd(DBG_ERROR, "Error packing arguments\n");
		assert(0); // FIXME Is there a better strategy to failing?
	}
	pkt->args[1].arr_argi[0] =
		getSize_regFuncArgs(fatCubinHandle, hostFun, deviceFun, deviceName,
				thread_limit, tid, bid, bDim, gDim, wSize);
	pkt->len = sizeof(*pkt) + pkt->args[1].arr_argi[0];
	pkt->is_sync = method_synctable[pkt->method_id];
}

static inline void
unpack_cudaRegisterFunction(struct cuda_packet *pkt, void *buf,
		reg_func_args_t *args)
{
	if (0 > unpackRegFuncArgs(args, (char*)((uintptr_t)buf + pkt->args[0].argull)))
		BUG(1);
}

static inline void
pack_cudaRegisterVar(struct cuda_packet *pkt, void *buf,
		void **fatCubinHandle,	//! cubin this var associates with
		char *hostVar,			//! addr of a var within app (not string)
		char *deviceAddress,	//! 8-byte device addr
		const char *deviceName, //! actual string
		int ext, int vsize, int constant, int global)
{
	int err;
	pkt->method_id = __CUDA_REGISTER_VARIABLE;
	pkt->thr_id = pthread_self();

	// now pack it into the shm
	pkt->args[0].argull = 0UL; // offset of variables in buf
	err = packRegVar(buf, fatCubinHandle, hostVar, deviceAddress, deviceName,
			ext, vsize, constant, global);
	if (err < 0) {
		printd(DBG_ERROR, "Error packing arguments\n");
		assert(0); // FIXME Is there a better strategy to failing?
	}
	pkt->args[1].arr_argi[0]
		= getSize_regVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
				ext, vsize, constant, global);
	pkt->len = sizeof(*pkt) + pkt->args[1].arr_argi[0];
	pkt->is_sync = method_synctable[pkt->method_id];

	// Add it to our list of known variable symbols.
	registered_vars[num_registered_vars++] = (uintptr_t)hostVar;
	if (num_registered_vars >= MAX_REGISTERED_VARS) BUG(1);
}

static inline void
unpack_cudaRegisterVar(struct cuda_packet *pkt, void *buf,
		reg_var_args_t *args)
{
	if (0 > unpackRegVar(args, (char *)((uintptr_t)buf + pkt->args[0].argull)))
		BUG(1);
}

// TODO
// 		cudaRegisterTexture

/*--------------------- TEMPLATES --------------*/

static inline void
pack_cuda(struct cuda_packet *pkt)
{
}

static inline void
extract_cuda(struct cuda_packet *pkt)
{
}

#endif /* MARSHAL_H_ */

/**
 * @file common/cuda/execute.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-22
 * @brief Execute serialized CUDA packets. CUDA packets are assumed to be
 * associated with a region of memory wherein any data relevant to the call may
 * lie (e.g. input or output pointer arguments). All of these functions execute
 * the methods in the CUDA Runtime API directly.
 */

// System includes
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>

// CUDA includes
#include <cuda_runtime_api.h>

// Project includes
#include <cuda/fatcubininfo.h>
#include <cuda/hidden.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <cuda/ops.h>
#include <cuda/packet.h>
#include <debug.h>
#include <util/compiler.h>
#include <util/timer.h>

/*-------------------------------------- NOTES -------------------------------*/

/*
 * The following functions shall be interposed by the assembly runtime and are
 * not made accessible elsewhere (including this file).
 *
 * 		cudaGetDevice
 * 		cudaGetDeviceCount
 * 		cudaGetDeviceProperties
 * 		cudaDriverGetVersion
 * 		cudaRuntimeGetVersion
 *
 * Some functions here require an additional argument; it is found at the
 * beginning of the va_list:
 *
 * 		struct fatcubins*
 *
 * Any required data must be accessible through the packet directly, and no
 * assumptions are made regarding the existance of assemblies, networking, etc.
 *
 * The only assumption that IS made, deals with the location of pointer
 * arguments. Currently every packet is assumed to accompany a memory region,
 * and that the packet itself lies at the top of this region. Arguments are
 * serialized into the packet according to the ordering within the function
 * prototype defined by the CUDA API itself. Arguments representing pointer data
 * (to be read or written to) are assumed to contain zero or postive-integer
 * values representing the offset the data can be found at in this memory
 * region. As we assume the packet itself sits at the top, its address is
 * combined with the specified offset to access the location of data for that
 * specific argument.
 */

// XXX XXX XXX
// Give the cuda packet a shm pointer instead of assuming it lies WITHIN the
// memory region itself. We can still place it in the memory region, but using a
// separate variable removes the assumption that it MUST be so.

/*-------------------------------------- MACROS ------------------------------*/

/**
 * Acquire the pointer to the cubins list from the variable argument list,
 * assuming it is the first argument in the list, and that no other args which
 * are needed inside a function can be found in the VA list. The second argument
 * to va_start must be the last named parameter specified in OPS_FN_PROTO, which
 * is specifed as the 'argname' to this macro. 'cubins' must be of type
 * struct fatcubins* which is modified to point to the actual list.
 */
#define GET_CUBIN_VALIST(cubins,argname)				\
	do {												\
		va_list extra;									\
		va_start(extra, argname);						\
		(cubins) = va_arg(extra, struct fatcubins*);	\
		va_end(extra);									\
	} while(0)

#define EXAMINE_CUDAERROR(pkt)									\
	do {														\
		if ((pkt)->ret_ex_val.err != cudaSuccess) {				\
			printd(DBG_ERROR, "returned non-success: %s\n",		\
					cudaGetErrorString((pkt)->ret_ex_val.err));	\
			BUG(1);												\
		}														\
	} while(0)

/*-------------------------------------- CUDA API ----------------------------*/

//
// Thread Management API
//

static OPS_FN_PROTO(CudaThreadExit)
{
	TIMER_DECLARE1(timer);
	printd(DBG_DEBUG, "thread exit\n");

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaThreadExit();
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaThreadSynchronize)
{
	TIMER_DECLARE1(timer);
	printd(DBG_DEBUG, "thread sync\n");

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaThreadSynchronize();
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Device Management API
//

// Many of these functions are intercepted by the assembly framework.

static OPS_FN_PROTO(CudaSetDevice)
{
	TIMER_DECLARE1(timer);
	int dev = pkt->args[0].argll;
	printd(DBG_DEBUG, "setDev %d\n", dev);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaSetDevice(dev);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetDeviceFlags)
{
	TIMER_DECLARE1(timer);
	unsigned int flags = (unsigned int)pkt->args[0].argull;
	printd(DBG_DEBUG, "flags=0x%x\n", flags);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaSetDeviceFlags(flags);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetValidDevices)
{
	TIMER_DECLARE1(timer);
	int *device_arr = (int*)((uintptr_t)pkt + pkt->args[0].argll);
	int len = pkt->args[1].argll;
	printd(DBG_DEBUG, "len=%d\n", len);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaSetValidDevices(device_arr, len);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Stream Management API
//

static OPS_FN_PROTO(CudaStreamCreate)
{
	TIMER_DECLARE1(timer);
	static unsigned long unique_bogus_stream_id = 0xdead00000000;
	printd(DBG_WARNING, "streams ignored in current implementation\n");

	TIMER_START(timer);
	// Until we do something meaningful with streams, just return a bogus value.
	// It's an opaque type to the caller, they won't care what the value is.
	pkt->args[0].stream = (cudaStream_t)unique_bogus_stream_id++;
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt); // not necessary at the moment
	return 0;
}

static OPS_FN_PROTO(CudaStreamSynchronize)
{
	TIMER_DECLARE1(timer);
	printd(DBG_WARNING, "streams ignored in current implementation\n");

	TIMER_START(timer);
	// ignore args[0].stream
	// call threadsync (does this simulate the behavior?)
	pkt->ret_ex_val.err = cudaThreadSynchronize();
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Execution Control API
//

static OPS_FN_PROTO(CudaConfigureCall)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	dim3 gridDim = pkt->args[0].arg_dim;
	dim3 blockDim = pkt->args[1].arg_dim;
	size_t sharedMem = pkt->args[2].arr_argi[0];
	cudaStream_t stream = (cudaStream_t)pkt->args[3].argull;

	printd(DBG_DEBUG, "grid={%u,%u,%u} block={%u,%u,%u}"
			" shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = 
		cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFuncGetAttributes)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *attr = (void*)((uintptr_t)pkt + pkt->args[0].argull); // output arg
	char *func = (char*)((uintptr_t)pkt + pkt->args[1].argull); // func name
	printd(DBG_DEBUG, "funcGetAttr func='%s'\n", func);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaFuncGetAttributes(attr, func);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaLaunch)
{
	TIMER_DECLARE1(timer);

	reg_func_args_t *func;

	TIMER_START(timer);
	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// 'entry' is some hostFun symbol pointer
	const char *entry = (const char *)pkt->args[0].argull;
	printd(DBG_DEBUG, "launch(%p)\n", entry);

	// Locate the func structure; we assume func names are unique across cubins.
	bool found = false;
	cubins_for_each_cubin(cubin_list, fatcubin) {
		cubin_for_each_function(fatcubin, func) {
			if (func->hostFun == entry) found = true;
			if (found) break;
		}
		if (found) break;
	}
	BUG(!found);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaLaunch(func->hostFun);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetupArgument)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	const void *arg = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t size = pkt->args[1].arr_argi[0];
	size_t offset = pkt->args[1].arr_argi[1];
	printd(DBG_DEBUG, "setupArg arg=%p size=%lu offset=%lu\n",
			arg, size, offset);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaSetupArgument(arg, size, offset);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Memory Management API
//

static OPS_FN_PROTO(CudaFree)
{
	TIMER_DECLARE1(timer);

	void *ptr = pkt->args[0].argp;
	printd(DBG_DEBUG, "free %p\n", ptr);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaFree(ptr);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFreeArray)
{
	TIMER_DECLARE1(timer);

	struct cudaArray *array = pkt->args[0].cudaArray;
	printd(DBG_DEBUG, "freeArray %p\n", array);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaFreeArray(array);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFreeHost)
{
	TIMER_DECLARE1(timer);

	void *ptr = pkt->args[0].argp;
	printd(DBG_DEBUG, "freeHost %p\n", ptr);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaFreeHost(ptr);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaHostAlloc)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *ptr;
	size_t size = pkt->args[1].arr_argi[0];
	unsigned int flags = pkt->args[2].argull;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaHostAlloc(&ptr, size, flags);
	pkt->args[0].argull = (uintptr_t)ptr;
	printd(DBG_DEBUG, "hostAlloc size=%lu flags=0x%x addr=%p\n",
			size, flags, ptr);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMalloc)
{
	TIMER_DECLARE1(timer);

	// We are to write the value of devPtr to args[0].argull
	void *devPtr = NULL;
	size_t size = pkt->args[1].arr_argi[0];

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMalloc(&devPtr, size);
	pkt->args[0].argull = (unsigned long long)devPtr;
	printd(DBG_DEBUG, "cudaMalloc devPtr=%p size=%lu ret:%u\n",
			devPtr, size, pkt->ret_ex_val.err);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMallocArray)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	struct cudaArray *array; // a handle for an array
	const struct cudaChannelFormatDesc *desc =
		(const struct cudaChannelFormatDesc*)
				((uintptr_t)pkt + pkt->args[0].argull);
	size_t width = pkt->args[1].arr_argi[0];
	size_t height = pkt->args[1].arr_argi[1];
	unsigned int flags = pkt->args[2].argull;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMallocArray(&array, desc, width, height, flags);
	pkt->args[0].cudaArray = array; // return value
	printd(DBG_DEBUG, "array=%p width=%lu height=%lu flags=0x%x\n",
			array, width, height, flags);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMallocPitch)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	// We are to write the value of devPtr to args[0].argull, and the value of
	// pitch to args[1].arr_argi[0].
	void *devPtr;
	size_t pitch;
	size_t width = pkt->args[2].arr_argi[0];
	size_t height = pkt->args[2].arr_argi[1];
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMallocPitch(&devPtr, &pitch, width, height);
	pkt->args[0].argull = (unsigned long long)devPtr;
	pkt->args[1].arr_argi[0] = pitch;
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "cudaMallocPitch devPtr=%p pitch=%lu"
			" width=%lu height=%lu ret:%u\n",
			devPtr, pitch, width, height, pkt->ret_ex_val.err);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyH2D)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *dst = (void*)pkt->args[0].argull; // gpu ptr
	const void *src = (const void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "memcpyh2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2H)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	void *src = (void*)pkt->args[1].argull; // gpu ptr
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "memcpyd2h dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2D)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *dst = (void*)pkt->args[0].argull;
	void *src = (void*)pkt->args[1].argull;
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "memcpyd2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyAsyncH2D)
{
	printd(DBG_WARNING, "async API not supported; defaulting to sync API\n");
	// we assume the arguments are the same, ignoring the stream
	return exec_ops.memcpyH2D(pkt);
}

static OPS_FN_PROTO(CudaMemcpyAsyncD2H)
{
	printd(DBG_WARNING, "async API not supported; defaulting to sync API\n");
	// we assume the arguments are the same, ignoring the stream
	return exec_ops.memcpyD2H(pkt);
}

static OPS_FN_PROTO(CudaMemcpyAsyncD2D)
{
	printd(DBG_WARNING, "async API not supported; defaulting to sync API\n");
	// we assume the arguments are the same, ignoring the stream
	return exec_ops.memcpyD2D(pkt);
}

static OPS_FN_PROTO(CudaMemcpyFromSymbolD2H)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];

	struct cuda_fatcubin_info *fatcubin;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// The application can provide symbols EITHER as a string literal containing
	// the name of the variable, OR the address directly. Thus 'symbol' will
	// either point to a string (within the shm region) or contain an address
	// representing the variable (which should not be dereferenced!).
	char *symbol = NULL;

	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		symbol = (char*)((uintptr_t)pkt + pkt->args[1].argull);
	} else {
		reg_var_args_t *var;
		symbol = (char*)(pkt->args[1].argull);
		// Locate the var structure; symbols are unique across cubins.
		// See quote from A. Kerr in libci.c
		// XXX TODO Move this code to a common function or something.
		bool found = false;
		cubins_for_each_cubin(cubins, fatcubin) {
			cubin_for_each_variable(fatcubin, var) {
				if (var->hostVar == symbol) found = true;
				if (found) break;
			}
			if (found) break;
		}
		BUG(!found);
		symbol = var->dom0HostAddr;
	}

	printd(DBG_DEBUG, "memcpyFromSymb symb=%p count=%lu\n", symbol, count);
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", symbol);
	}
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err =
		cudaMemcpyFromSymbol(dst, symbol, count, offset,
				cudaMemcpyDeviceToHost);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToArrayH2D)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	struct cudaArray *dst = pkt->args[0].cudaArray;
	size_t wOffset = pkt->args[1].arr_argi[0];
	size_t hOffset = pkt->args[1].arr_argi[1];
	const void *src = (const void*)((uintptr_t)pkt + pkt->args[2].argull);
	size_t count = pkt->args[3].arr_argi[0];
	printd(DBG_DEBUG, "called\n");
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err =
		cudaMemcpyToArray(dst, wOffset, hOffset,
				src, count, cudaMemcpyHostToDevice);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToArrayD2D)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	struct cudaArray *dst = pkt->args[0].cudaArray;
	size_t wOffset = pkt->args[1].arr_argi[0];
	size_t hOffset = pkt->args[1].arr_argi[1];
	const void *src = pkt->args[2].argp; // gpu ptr?
	size_t count = pkt->args[3].arr_argi[0];
	printd(DBG_DEBUG, "called\n");
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err =
		cudaMemcpyToArray(dst, wOffset, hOffset,
				src, count, cudaMemcpyDeviceToDevice);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToSymbolH2D)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	const void *src = (void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// See comments in CudaMemcpyFromSymbolD2H.

	char *symbol = NULL;

	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		symbol = (char*)((uintptr_t)pkt + pkt->args[0].argull);
	} else {
		reg_var_args_t *var = NULL;
		symbol = (char*)(pkt->args[0].argull);
		bool found = false;
		cubins_for_each_cubin(cubins, fatcubin) {
			cubin_for_each_variable(fatcubin, var) {
				if (var->hostVar == symbol) found = true;
				if (found) break;
			}
			if (found) break;
		}
		BUG(!found);
		symbol = var->dom0HostAddr;
	}

	printd(DBG_DEBUG, "memcpyFromSymb symb=%p count=%lu\n", symbol, count);
	if (pkt->flags & CUDA_PKT_SYMB_IS_STRING) {
		printd(DBG_DEBUG, "\tsymbol is string: %s\n", symbol);
	}
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err =
		cudaMemcpyToSymbol(symbol, src, count, offset,
				cudaMemcpyHostToDevice);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToSymbolAsyncH2D)
{
	printd(DBG_WARNING, "async API not supported; defaulting to sync API\n");
	// we assume the arguments are the same, ignoring the stream
	return exec_ops.memcpyToSymbolH2D(pkt);
}

static OPS_FN_PROTO(CudaMemGetInfo)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	size_t *free, *total;
	free = &(pkt->args[0].arr_argi[0]);
	total = &(pkt->args[0].arr_argi[1]);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMemGetInfo(free, total);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "memGetInfo free=%lu total=%lu\n", *free, *total);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemset)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	void *devPtr = (void*)pkt->args[0].argull;
	int value = pkt->args[1].argll;
	size_t count = pkt->args[2].arr_argi[0];
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaMemset(devPtr, value, count);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "cudaMemset devPtr=%p value=%d count=%lu\n",
			devPtr, value, count);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Texture Management API
//

static OPS_FN_PROTO(CudaBindTexture)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	struct textureReference *texRef = pkt->args[0].argp; // symbol addr
	struct textureReference *new_tex = &pkt->args[1].texRef; // data at symbol
	void *devPtr = pkt->args[2].argp;
	struct cudaChannelFormatDesc *desc = &pkt->args[3].desc;
	size_t size = pkt->args[4].arr_argi[0];
	size_t offset;
	reg_tex_args_t *tex;

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// Look for the texture. Associate the application reference with the one
	// registered in the sink address space.
	bool found = false;
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_texture(fatcubin, tex) {
			if (tex->texRef == texRef) found = true;
			if (found) break;
		}
		if (found) break;
	}
	BUG(!found);

	// Copy the structure to the locally maintained version, as the caller may
	// have updated the values in its copy of the structure before invoking this
	// routine.
	tex->tex = *new_tex;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err = cudaBindTexture(&offset, &tex->tex, devPtr, desc, size);
	pkt->args[0].arr_argi[0] = offset;
	TIMER_END(timer, pkt->lat.exec.call);
	return 0;
}

static OPS_FN_PROTO(CudaBindTextureToArray)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	//! address of texture variable in application; use for lookup
	struct textureReference *texRef = pkt->args[0].argp;
	//! new state of texture app has provided
	struct textureReference *new_tex = &pkt->args[1].texRef;
	struct cudaArray *array = pkt->args[2].cudaArray;
	//! data describing channel format
	struct cudaChannelFormatDesc *desc = &pkt->args[3].desc;
	reg_tex_args_t *tex;

	printd(DBG_DEBUG, "called\n");

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// Look for the texture. Associate the application reference with the one
	// registered in the sink address space.
	bool found = false;
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_texture(fatcubin, tex) {
			if (tex->texRef == texRef) found = true;
			if (found) break;
		}
		if (found) break;
	}
	BUG(!found);

	// Copy the structure to the locally maintained version, as the caller may
	// have updated the values in its copy of the structure before invoking this
	// routine.
	tex->tex = *new_tex;
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->ret_ex_val.err =
		cudaBindTextureToArray(&tex->tex, array, desc);
	TIMER_END(timer, pkt->lat.exec.call);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaCreateChannelDesc)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	int x = pkt->args[0].arr_argii[0];
	int y = pkt->args[0].arr_argii[1];
	int z = pkt->args[0].arr_argii[2];
	int w = pkt->args[0].arr_argii[3];
	enum cudaChannelFormatKind format = pkt->args[1].arr_arguii[0];
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u\n", x,y,z,w,format);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	pkt->args[0].desc = cudaCreateChannelDesc(x,y,z,w,format);
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);
	return 0;
}

static OPS_FN_PROTO(CudaGetTextureReference)
{
	//const struct textureReference *texRef = NULL; // runtime modifies this
	//char *symbol = (char *)(pkt->args[0].argull);
	//pkt->ret_ex_val.err = cudaGetTextureReference(&texRef, symbol);
	//pkt->args[0].argull = (uintptr_t)*texRef;
	//EXAMINE_CUDAERROR(pkt);
	BUG(0);
	return 0;
}

//
// Undocumented API
//

static OPS_FN_PROTO(__CudaRegisterFatBinary)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	int err;
	void *handle;
	void *cubin_shm = (void*)((uintptr_t)pkt + pkt->args[0].argull);

	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// Allocate space to store the new CUBIN and unmarshal it.
	__cudaFatCudaBinary *cuda_cubin =
		calloc(1, sizeof(__cudaFatCudaBinary));
	if (!cuda_cubin) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	err = unpackFatBinary(cuda_cubin, cubin_shm);
	if (err < 0) {
		printd(DBG_ERROR, "error unpacking fat cubin\n");
		goto fail;
	}
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	// Make the call, then add it to the cubin list.
	handle = __cudaRegisterFatBinary(cuda_cubin);
	pkt->ret_ex_val.handle = handle;
	cubins_add_cubin(cubin_list, cuda_cubin, handle);
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "local cudaRegFB(%p) returned %p\n",
			cuda_cubin, pkt->ret_ex_val.handle);
	return 0;

fail:
	return -1;
}

static OPS_FN_PROTO(__CudaUnregisterFatBinary)
{
	TIMER_DECLARE1(timer);

	void **handle = pkt->args[0].argdp;

	TIMER_START(timer);
	__cudaUnregisterFatBinary(handle);
	// FIXME Deallocate the fat binary data structures.
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "unregister FB handle=%p\n", handle);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(__CudaRegisterFunction)
{
	TIMER_DECLARE1(timer);
	int err, exit_errno;

	TIMER_START(timer);
	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// unpack the serialized arguments from shared mem
	reg_func_args_t *pargs =  // packed
		(reg_func_args_t*)((uintptr_t)pkt + pkt->args[0].argull);
	reg_func_args_t *uargs = // unpacked
		calloc(1, sizeof(reg_func_args_t));
	if (!uargs) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	err = unpackRegFuncArgs(uargs, (char *)pargs); // FIXME don't use char*
	if (err < 0) {
		printd(DBG_ERROR, "error unpacking regfunc args\n");
		exit_errno = -EPROTO;
		goto fail;
	}
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	// Make the call
	__cudaRegisterFunction(uargs->fatCubinHandle,
			(const char *)uargs->hostFun, uargs->deviceFun,
			(const char *)uargs->deviceName, uargs->thread_limit,
			uargs->tid, uargs->bid, uargs->bDim, uargs->gDim,
			uargs->wSize);

	// store the state
	cubins_add_function(cubin_list, uargs->fatCubinHandle, &uargs->link);
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "regFunc handle=%p\n", uargs->fatCubinHandle);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(__CudaRegisterVar)
{
	TIMER_DECLARE1(timer);

	TIMER_START(timer);
	int err, exit_errno;

	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// unpack the serialized arguments from shared mem
	reg_var_args_t *pargs = // packed
		(reg_var_args_t*)((uintptr_t)pkt + pkt->args[0].argull);
	reg_var_args_t *uargs = // unpacked
		calloc(1, sizeof(reg_var_args_t));
	if (!uargs) {
		fprintf(stderr, "Out of memory\n");
		exit_errno = -ENOMEM;
		goto fail;
	}
	err = unpackRegVar(uargs, (char *)pargs);
	if (err < 0) {
		printd(DBG_ERROR, "error unpacking regvar args\n");
		exit_errno = -EPROTO;
		goto fail;
	}
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	// Make the call
	__cudaRegisterVar(uargs->fatCubinHandle, uargs->dom0HostAddr,
			uargs->deviceAddress, (const char *) uargs->deviceName,
			uargs->ext, uargs->size, uargs->constant, uargs->global);

	// store the state
	cubins_add_variable(cubin_list, uargs->fatCubinHandle, &uargs->link);
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);

	printd(DBG_DEBUG, "_regVar: %p\n", uargs->hostVar);
	EXAMINE_CUDAERROR(pkt);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(__CudaRegisterTexture)
{
	TIMER_DECLARE1(timer);
	int exit_errno;

	TIMER_START(timer);
	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);
	
	char *texName = (char*)((uintptr_t)pkt + pkt->args[4].argull);
	printd(DBG_DEBUG, "name='%s' name_len=%lu\n", texName, strlen(texName) + 1);

	reg_tex_args_t *tex = calloc(1, sizeof(*tex));
	if (!tex) {
		fprintf(stderr, "Out of memory\n");
		printd(DBG_ERROR, "Out of memory\n");
		exit_errno = -ENOMEM;
		goto fail;
	}
	INIT_LIST_HEAD(&tex->link);
	// TODO Move all this packing out of this function
	tex->fatCubinHandle = pkt->args[0].argdp; // pointer copy
	tex->texRef = pkt->args[1].argp; // address of variable in application
	tex->tex = pkt->args[2].texRef; // initial state
	tex->devPtr = pkt->args[3].argp; // pointer copy
	tex->texName = calloc(1, strlen(texName) * 2);
	strncpy((char *)tex->texName, texName, strlen(texName) + 1);
	tex->dim = pkt->args[5].arr_argii[0];
	tex->norm = pkt->args[5].arr_argii[1];
	tex->ext = pkt->args[5].arr_argii[2];

	printd(DBG_DEBUG, "handle=%p devPtr=%p texName=%s dim=%d norm=%d ext=%d\n",
			tex->fatCubinHandle, tex->devPtr, tex->texName,
			tex->dim, tex->norm, tex->ext);
	TIMER_END(timer, pkt->lat.exec.setup);

	TIMER_START(timer);
	// Make the call
	__cudaRegisterTexture(tex->fatCubinHandle,
			&tex->tex, // pretend this is our global; register its addr
			&tex->devPtr, tex->texName,
			tex->dim, tex->norm, tex->ext);

	// store the state
	cubins_add_texture(cubin_list, tex->fatCubinHandle, &tex->link);
	pkt->ret_ex_val.err = cudaSuccess;
	TIMER_END(timer, pkt->lat.exec.call);
	return 0;

fail:
	return exit_errno;
}

const struct cuda_ops exec_ops =
{
	// Functions which take only a cuda_packet*
	.bindTexture = CudaBindTexture,
	.configureCall = CudaConfigureCall,
	.createChannelDesc = CudaCreateChannelDesc,
	.freeArray = CudaFreeArray,
	.free = CudaFree,
	.freeHost = CudaFreeHost,
	.funcGetAttributes = CudaFuncGetAttributes,
	.getTextureReference = CudaGetTextureReference,
	.hostAlloc = CudaHostAlloc,
	.mallocArray = CudaMallocArray,
	.malloc = CudaMalloc,
	.mallocPitch = CudaMallocPitch,
	.memcpyAsyncD2D = CudaMemcpyAsyncD2D,
	.memcpyAsyncD2H = CudaMemcpyAsyncD2H,
	.memcpyAsyncH2D = CudaMemcpyAsyncH2D,
	.memcpyD2D = CudaMemcpyD2D,
	.memcpyD2H = CudaMemcpyD2H,
	.memcpyH2D = CudaMemcpyH2D,
	.memcpyToArrayD2D = CudaMemcpyToArrayD2D,
	.memcpyToArrayH2D = CudaMemcpyToArrayH2D,
	.memcpyToSymbolAsyncH2D = CudaMemcpyToSymbolAsyncH2D,
	.memGetInfo = CudaMemGetInfo,
	.memset = CudaMemset,
	.registerTexture = __CudaRegisterTexture,
	.setDevice = CudaSetDevice,
	.setDeviceFlags = CudaSetDeviceFlags,
	.setupArgument = CudaSetupArgument,
	.setValidDevices = CudaSetValidDevices,
	.streamCreate = CudaStreamCreate,
	.streamSynchronize = CudaStreamSynchronize,
	.threadExit = CudaThreadExit,
	.threadSynchronize = CudaThreadSynchronize,
	.unregisterFatBinary = __CudaUnregisterFatBinary,

	// Functions which take a cuda_packet* and fatcubins*
	.bindTextureToArray = CudaBindTextureToArray,
	.launch = CudaLaunch,
	.memcpyFromSymbolD2H = CudaMemcpyFromSymbolD2H,
	.memcpyToSymbolH2D = CudaMemcpyToSymbolH2D,
	.registerFatBinary = __CudaRegisterFatBinary,
	.registerFunction = __CudaRegisterFunction,
	.registerVar = __CudaRegisterVar,
};

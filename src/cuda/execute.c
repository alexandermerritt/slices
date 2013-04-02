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
		if (unlikely((pkt)->ret_ex_val.err != cudaSuccess)) {	\
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
	printd(DBG_DEBUG, "thread exit\n");

	pkt->ret_ex_val.err = cudaThreadExit();

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaThreadSynchronize)
{
	printd(DBG_DEBUG, "thread sync\n");

	pkt->ret_ex_val.err = cudaThreadSynchronize();

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Device Management API
//

// Many of these functions are intercepted by the assembly framework.

static OPS_FN_PROTO(CudaSetDevice)
{
	int dev;
	unpack_cudaSetDevice(pkt, &dev);
	printd(DBG_DEBUG, "setDev %d\n", dev);

	pkt->ret_ex_val.err = cudaSetDevice(dev);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetDeviceFlags)
{
	unsigned int flags;
	unpack_cudaSetDeviceFlags(pkt, &flags);
	printd(DBG_DEBUG, "flags=0x%x\n", flags);

	pkt->ret_ex_val.err = cudaSetDeviceFlags(flags);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetValidDevices)
{
	int *device_arr;
	int len;
	unpack_cudaSetValidDevices(pkt, (pkt + 1), &device_arr, &len);
	printd(DBG_DEBUG, "len=%d\n", len);

	pkt->ret_ex_val.err = cudaSetValidDevices(device_arr, len);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Stream Management API
//

static OPS_FN_PROTO(CudaStreamCreate)
{
	cudaStream_t stream;

	pkt->ret_ex_val.err = cudaStreamCreate(&stream);
	insert_cudaStreamCreate(pkt, stream);

	printd(DBG_DEBUG, "stream=%p ret=%u\n", stream, pkt->ret_ex_val.err);

	EXAMINE_CUDAERROR(pkt); // not necessary at the moment
	return 0;
}

static OPS_FN_PROTO(CudaStreamDestroy)
{
	cudaStream_t stream;

	unpack_cudaStreamDestroy(pkt, &stream);

	pkt->ret_ex_val.err = cudaStreamDestroy(stream);

	printd(DBG_DEBUG, "stream=%p ret=%u\n", stream, pkt->ret_ex_val.err);

	EXAMINE_CUDAERROR(pkt); // not necessary at the moment
	return 0;
}

static OPS_FN_PROTO(CudaStreamQuery)
{
	cudaStream_t stream;

	unpack_cudaStreamQuery(pkt, &stream);

	pkt->ret_ex_val.err = cudaStreamQuery(stream);

	printd(DBG_DEBUG, "stream=%p ret=%u\n", stream, pkt->ret_ex_val.err);

	EXAMINE_CUDAERROR(pkt); // not necessary at the moment
	return 0;
}

static OPS_FN_PROTO(CudaStreamSynchronize)
{
	cudaStream_t stream;

	unpack_cudaStreamSynchronize(pkt, &stream);

	pkt->ret_ex_val.err = cudaStreamSynchronize(stream);

	printd(DBG_DEBUG, "stream=%p ret=%u\n", stream, pkt->ret_ex_val.err);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Event Management API
//

static OPS_FN_PROTO(CudaEventCreate)
{
    cudaEvent_t event;
    pkt->ret_ex_val.err = cudaEventCreate(&event);
    printd(DBG_DEBUG, "event=%p ret=%u\n", event, pkt->ret_ex_val.err);
    insert_cudaEventCreate(pkt, event);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventCreateWithFlags)
{
    cudaEvent_t event;
    unsigned int flags;
    unpack_cudaEventCreateWithFlags(pkt, &flags);
    pkt->ret_ex_val.err = cudaEventCreateWithFlags(&event, flags);
    printd(DBG_DEBUG, "flags=%x event=%p ret=%u\n",
            flags, event, pkt->ret_ex_val.err);
    insert_cudaEventCreateWithFlags(pkt, event);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventDestroy)
{
    cudaEvent_t event;
    unpack_cudaEventDestroy(pkt, &event);
    pkt->ret_ex_val.err = cudaEventDestroy(event);
    printd(DBG_DEBUG, "event=%p ret=%u\n", event, pkt->ret_ex_val.err);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventElapsedTime)
{
    float ms;
    cudaEvent_t start, end;
    unpack_cudaEventElapsedTime(pkt, &start, &end);
    pkt->ret_ex_val.err = cudaEventElapsedTime(&ms, start, end);
    insert_cudaEventElapsedTime(pkt, ms);
    printd(DBG_DEBUG, "start=%p end=%p ms=%f ret=%u\n",
            start, end, ms, pkt->ret_ex_val.err);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventQuery)
{
    cudaEvent_t event;
    unpack_cudaEventQuery(pkt, &event);
    pkt->ret_ex_val.err = cudaEventQuery(event);
    printd(DBG_DEBUG, "event=%p ret=%u\n", event, pkt->ret_ex_val.err);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventRecord)
{
    cudaEvent_t event;
    cudaStream_t stream;
    unpack_cudaEventRecord(pkt, &event, &stream);
    pkt->ret_ex_val.err = cudaEventRecord(event, stream);
    printd(DBG_DEBUG, "event=%p stream=%p ret=%u\n",
            event, stream, pkt->ret_ex_val.err);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

static OPS_FN_PROTO(CudaEventSynchronize)
{
    cudaEvent_t event;
    unpack_cudaEventSynchronize(pkt, &event);
    pkt->ret_ex_val.err = cudaEventSynchronize(event);
    printd(DBG_DEBUG, "event=%p ret=%u\n", event, pkt->ret_ex_val.err);
    EXAMINE_CUDAERROR(pkt);
    return 0;
}

//
// Execution Control API
//

static OPS_FN_PROTO(CudaConfigureCall)
{

	dim3 gridDim, blockDim;
	size_t sharedMem;
	cudaStream_t stream;
	unpack_cudaConfigureCall(pkt, &gridDim, &blockDim, &sharedMem, &stream);

	printd(DBG_DEBUG, "grid={%u,%u,%u} block={%u,%u,%u}"
			" shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

	pkt->ret_ex_val.err = 
		cudaConfigureCall(gridDim, blockDim, sharedMem, stream);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFuncGetAttributes)
{

	void *attr = (void*)((uintptr_t)pkt + pkt->args[0].argull); // output arg
	char *func = (char*)((uintptr_t)pkt + pkt->args[1].argull); // func name
	printd(DBG_DEBUG, "funcGetAttr func='%s'\n", func);

	pkt->ret_ex_val.err = cudaFuncGetAttributes(attr, func);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaLaunch)
{

	reg_func_args_t *func;

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// 'entry' is some hostFun symbol pointer
	const char *entry;
	unpack_cudaLaunch(pkt, &entry);
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

	pkt->ret_ex_val.err = cudaLaunch(func->hostFun);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaSetupArgument)
{

	const void *arg;
	size_t size;
	size_t offset;
	unpack_cudaSetupArgument(pkt, (pkt + 1), &arg, &size, &offset);
	printd(DBG_DEBUG, "setupArg arg=%p size=%lu offset=%lu\n",
			arg, size, offset);

	pkt->ret_ex_val.err = cudaSetupArgument(arg, size, offset);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

//
// Memory Management API
//

static OPS_FN_PROTO(CudaFree)
{

	void *ptr;
	unpack_cudaFree(pkt, &ptr);
	printd(DBG_DEBUG, "free %p\n", ptr);

	pkt->ret_ex_val.err = cudaFree(ptr);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFreeArray)
{

	struct cudaArray *array;
	unpack_cudaFreeArray(pkt, &array);
	printd(DBG_DEBUG, "freeArray %p\n", array);

	pkt->ret_ex_val.err = cudaFreeArray(array);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaFreeHost)
{

	void *ptr = pkt->args[0].argp;
	printd(DBG_DEBUG, "freeHost %p\n", ptr);

	pkt->ret_ex_val.err = cudaFreeHost(ptr);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaHostAlloc)
{

	void *ptr;
	size_t size = pkt->args[1].arr_argi[0];
	unsigned int flags = pkt->args[2].argull;

	pkt->ret_ex_val.err = cudaHostAlloc(&ptr, size, flags);
	pkt->args[0].argull = (uintptr_t)ptr;
	printd(DBG_DEBUG, "hostAlloc size=%lu flags=0x%x addr=%p\n",
			size, flags, ptr);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMalloc)
{

	// We are to write the value of devPtr to args[0].argull
	void *devPtr = NULL;
	size_t size;
	unpack_cudaMalloc(pkt, &size);

	pkt->ret_ex_val.err = cudaMalloc(&devPtr, size);
	insert_cudaMalloc(pkt, devPtr);
	printd(DBG_DEBUG, "cudaMalloc devPtr=%p size=%lu ret:%u\n",
			devPtr, size, pkt->ret_ex_val.err);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMallocArray)
{

	struct cudaArray *array; // a handle for an array
	struct cudaChannelFormatDesc *desc;
	size_t width;
	size_t height;
	unsigned int flags;
	unpack_cudaMallocArray(pkt, &desc, &width, &height, &flags);

	pkt->ret_ex_val.err = cudaMallocArray(&array, desc, width, height, flags);
	insert_cudaMallocArray(pkt, array);
	printd(DBG_DEBUG, "array=%p width=%lu height=%lu flags=0x%x\n",
			array, width, height, flags);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMallocPitch)
{

	// We are to write the value of devPtr to args[0].argull, and the value of
	// pitch to args[1].arr_argi[0].
	void *devPtr;
	size_t pitch;
	size_t width;
	size_t height;
	unpack_cudaMallocPitch(pkt, &width, &height);

	pkt->ret_ex_val.err = cudaMallocPitch(&devPtr, &pitch, width, height);
	insert_cudaMallocPitch(pkt, devPtr, pitch);

	printd(DBG_DEBUG, "cudaMallocPitch devPtr=%p pitch=%lu"
			" width=%lu height=%lu ret:%u\n",
			devPtr, pitch, width, height, pkt->ret_ex_val.err);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyH2D)
{

	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
	unpack_cudaMemcpy(pkt, (pkt + 1), &dst, &src, &count, kind);

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);

	printd(DBG_DEBUG, "memcpyh2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2H)
{

	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
	unpack_cudaMemcpy(pkt, (pkt + 1), &dst, &src, &count, kind);

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);

	printd(DBG_DEBUG, "memcpyd2h dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2D)
{

	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
	unpack_cudaMemcpy(pkt, (pkt + 1), &dst, &src, &count, kind);

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);

	printd(DBG_DEBUG, "memcpyd2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpy2DH2D)
{
	void *dst;
	const void *src;
	size_t count, dpitch, spitch, width, height;
	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
	unpack_cudaMemcpy2D(pkt, (pkt + 1),
            &dst, &dpitch, &src, &spitch, &width, &height, kind);

    count = (width * height);
	pkt->ret_ex_val.err =
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

	printd(DBG_DEBUG, "memcpy2Dh2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpy2DD2H)
{
	void *dst;
	const void *src;
	size_t count, dpitch, spitch, width, height;
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
	unpack_cudaMemcpy2D(pkt, (pkt + 1),
            &dst, &dpitch, &src, &spitch, &width, &height, kind);

    count = (width * height);
	pkt->ret_ex_val.err =
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

	printd(DBG_DEBUG, "memcpy2Dd2h dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpy2DD2D)
{
	void *dst;
	const void *src;
	size_t count, dpitch, spitch, width, height;
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
	unpack_cudaMemcpy2D(pkt, (pkt + 1),
            &dst, &dpitch, &src, &spitch, &width, &height, kind);

    count = (width * height);
	pkt->ret_ex_val.err =
        cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);

	printd(DBG_DEBUG, "memcpy2Dd2d dst=%p src=%p count=%lu kind=%u\n",
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

	void *dst;
	size_t count, offset;

	struct cuda_fatcubin_info *fatcubin;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// The application can provide symbols EITHER as a string literal containing
	// the name of the variable, OR the address directly. Thus 'symbol' will
	// either point to a string (within the shm region) or contain an address
	// representing the variable (which should not be dereferenced!).
	const char *symbol = NULL;

	unpack_cudaMemcpyFromSymbol(pkt, (pkt + 1),
			&dst, &symbol, &count, &offset, cudaMemcpyDeviceToHost);

	// Locate the var structure; symbols are unique across cubins.
	// See quote from A. Kerr in libci.c
	// XXX TODO Move this code to a common function or something.
	bool found = false;
	reg_var_args_t *var;
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_variable(fatcubin, var) {
			if (var->hostVar == symbol) found = true;
			if (found) break;
		}
		if (found) break;
	}
	BUG(!found);
	symbol = var->dom0HostAddr;


	pkt->ret_ex_val.err =
		cudaMemcpyFromSymbol(dst, symbol, count, offset,
				cudaMemcpyDeviceToHost);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToArrayH2D)
{

	struct cudaArray *dst;
	size_t wOffset;
	size_t hOffset;
	const void *src;
	size_t count;
	unpack_cudaMemcpyToArray(pkt, (pkt + 1),
			&dst, &wOffset, &hOffset, &src, &count, cudaMemcpyHostToDevice);
	printd(DBG_DEBUG, "called\n");

	pkt->ret_ex_val.err =
		cudaMemcpyToArray(dst, wOffset, hOffset,
				src, count, cudaMemcpyHostToDevice);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToArrayD2D)
{

	struct cudaArray *dst;
	size_t wOffset;
	size_t hOffset;
	const void *src;
	size_t count;
	unpack_cudaMemcpyToArray(pkt, (pkt + 1),
			&dst, &wOffset, &hOffset, &src, &count, cudaMemcpyHostToDevice);
	printd(DBG_DEBUG, "called\n");

	pkt->ret_ex_val.err =
		cudaMemcpyToArray(dst, wOffset, hOffset,
				src, count, cudaMemcpyDeviceToDevice);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToSymbolH2D)
{

	const void *src;
	size_t count, offset;
	const char *symbol = NULL;
	reg_var_args_t *var = NULL;
	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	unpack_cudaMemcpyToSymbol(pkt, (pkt + 1),
			&symbol, &src, &count, &offset, cudaMemcpyHostToDevice);
	// See comments in CudaMemcpyFromSymbolD2H.
	// Symbols which have not been registered via cudaRegisterVar are strings,
	// and don't require the below 'lookup'.
	if (!(pkt->flags & CUDA_PKT_SYMB_IS_STRING)) {
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

	pkt->ret_ex_val.err =
		cudaMemcpyToSymbol(symbol, src, count, offset,
				cudaMemcpyHostToDevice);

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
	size_t free, total;

	pkt->ret_ex_val.err = cudaMemGetInfo(&free, &total);
    insert_cudaMemGetInfo(pkt, free, total);

	printd(DBG_DEBUG, "memGetInfo free=%lu total=%lu\n", free, total);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaMemset)
{

	void *devPtr;
	int value;
	size_t count;

	unpack_cudaMemset(pkt, &devPtr, &value, &count);

	pkt->ret_ex_val.err = cudaMemset(devPtr, value, count);

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

	const struct textureReference *texRef; // symbol addr
	const struct textureReference *new_tex; // data at symbol
	const void *devPtr;
	const struct cudaChannelFormatDesc *desc;
	size_t size;
	size_t offset;
	reg_tex_args_t *tex;
	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;

	GET_CUBIN_VALIST(cubins, pkt);

	unpack_cudaBindTexture(pkt, &texRef, &new_tex, &devPtr, &desc, &size);

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

	pkt->ret_ex_val.err = cudaBindTexture(&offset, &tex->tex, devPtr, desc, size);
	insert_cudaBindTexture(pkt, offset);
	return 0;
}

static OPS_FN_PROTO(CudaBindTextureToArray)
{

	//! address of texture variable in application; use for lookup
	const struct textureReference *texRef;
	//! new state of texture app has provided
	const struct textureReference *new_tex;
	const struct cudaArray *array;
	//! data describing channel format
	const struct cudaChannelFormatDesc *desc;
	reg_tex_args_t *tex;

	printd(DBG_DEBUG, "called\n");

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;

	unpack_cudaBindTextureToArray(pkt, &texRef, &new_tex, &array, &desc);
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

	pkt->ret_ex_val.err =
		cudaBindTextureToArray(&tex->tex, array, desc);

	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(CudaCreateChannelDesc)
{

	int x = pkt->args[0].arr_argii[0];
	int y = pkt->args[0].arr_argii[1];
	int z = pkt->args[0].arr_argii[2];
	int w = pkt->args[0].arr_argii[3];
	enum cudaChannelFormatKind format = (enum cudaChannelFormatKind)pkt->args[1].arr_arguii[0];
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u\n", x,y,z,w,format);

	pkt->args[0].desc = cudaCreateChannelDesc(x,y,z,w,format);
	pkt->ret_ex_val.err = cudaSuccess;
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

	void *handle;

	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// Allocate space to store the new CUBIN and unmarshal it.
	__cudaFatCudaBinary *cuda_cubin = calloc(1, sizeof(__cudaFatCudaBinary));
	if (!cuda_cubin) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	unpack_cudaRegisterFatBinary(pkt, (pkt + 1), cuda_cubin);

	// Make the call, then add it to the cubin list.
	handle = __cudaRegisterFatBinary(cuda_cubin);
	pkt->ret_ex_val.handle = handle;
	cubins_add_cubin(cubin_list, cuda_cubin, handle);

	printd(DBG_DEBUG, "local cudaRegFB(%p) returned %p\n",
			cuda_cubin, pkt->ret_ex_val.handle);
	return 0;

fail:
	return -1;
}

static OPS_FN_PROTO(__CudaUnregisterFatBinary)
{

	void **handle;

	unpack_cudaUnregisterFatBinary(pkt, &handle);
	__cudaUnregisterFatBinary(handle);
	// FIXME Deallocate the fat binary data structures.
	pkt->ret_ex_val.err = cudaSuccess;

	printd(DBG_DEBUG, "unregister FB handle=%p\n", handle);
	EXAMINE_CUDAERROR(pkt);
	return 0;
}

static OPS_FN_PROTO(__CudaRegisterFunction)
{
	int exit_errno;
	struct fatcubins *cubin_list = NULL;

	GET_CUBIN_VALIST(cubin_list, pkt);

	// unpack the serialized arguments from shared mem
	reg_func_args_t *uargs = calloc(1, sizeof(reg_func_args_t));
	if (!uargs) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	unpack_cudaRegisterFunction(pkt, (pkt + 1), uargs);

	printd(DBG_DEBUG, "handle=%p hostFun=%p deviceFun=%s deviceName=%s\n",
			uargs->fatCubinHandle, uargs->hostFun, uargs->deviceFun, uargs->deviceName);

	// Make the call
	__cudaRegisterFunction(uargs->fatCubinHandle,
			(const char *)uargs->hostFun, uargs->deviceFun,
			(const char *)uargs->deviceName, uargs->thread_limit,
			uargs->tid, uargs->bid, uargs->bDim, uargs->gDim,
			uargs->wSize);

	// store the state
	cubins_add_function(cubin_list, uargs->fatCubinHandle, &uargs->link);
	pkt->ret_ex_val.err = cudaSuccess;

	printd(DBG_DEBUG, "regFunc handle=%p\n", uargs->fatCubinHandle);
	return 0;

fail:
	return exit_errno;
}

// This registers user-level addresses of application global variables. If a
// symbol passed into a function that accepts a symbol is not found
static OPS_FN_PROTO(__CudaRegisterVar)
{

	int exit_errno;

	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// unpack the serialized arguments from shared mem
	reg_var_args_t *uargs = // unpacked
		calloc(1, sizeof(reg_var_args_t));
	if (!uargs) {
		fprintf(stderr, "Out of memory\n");
		exit_errno = -ENOMEM;
		goto fail;
	}
	unpack_cudaRegisterVar(pkt, (pkt + 1), uargs);

	// Make the call
	__cudaRegisterVar(uargs->fatCubinHandle, uargs->dom0HostAddr,
			uargs->deviceAddress, (const char *) uargs->deviceName,
			uargs->ext, uargs->size, uargs->constant, uargs->global);

	// store the state
	cubins_add_variable(cubin_list, uargs->fatCubinHandle, &uargs->link);
	pkt->ret_ex_val.err = cudaSuccess;

	printd(DBG_DEBUG, "_regVar: %p\n", uargs->hostVar);
	EXAMINE_CUDAERROR(pkt);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(__CudaRegisterTexture)
{
	int exit_errno;

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

	// Make the call
	__cudaRegisterTexture(tex->fatCubinHandle,
			&tex->tex, // pretend this is our global; register its addr
			&tex->devPtr, tex->texName,
			tex->dim, tex->norm, tex->ext);

	// store the state
	cubins_add_texture(cubin_list, tex->fatCubinHandle, &tex->link);
	pkt->ret_ex_val.err = cudaSuccess;
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

	.eventCreate            = CudaEventCreate,
	.eventCreateWithFlags   = CudaEventCreateWithFlags,
	.eventDestroy           = CudaEventDestroy,
	.eventElapsedTime       = CudaEventElapsedTime,
	.eventQuery             = CudaEventQuery,
	.eventRecord            = CudaEventRecord,
	.eventSynchronize       = CudaEventSynchronize,

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
	.memcpy2DD2D = CudaMemcpy2DD2D,
	.memcpy2DD2H = CudaMemcpy2DD2H,
	.memcpy2DH2D = CudaMemcpy2DH2D,
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
	.streamDestroy = CudaStreamDestroy,
	.streamQuery = CudaStreamQuery,
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

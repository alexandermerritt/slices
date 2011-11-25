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

// CUDA includes
#include <cuda_runtime_api.h>

// Project includes
#include <cuda_hidden.h>
#include <cuda/ops.h>
#include <debug.h>
#include <fatcubininfo.h>
// FIXME Remove this dependence. Move the marshaling code to another file,
// perhaps common/cuda/marshal.c (even if only the interposer uses it).
#include <libciutils.h>
#include <method_id.h>
#include <packetheader.h>
#include <util/compiler.h>

/* NOTES
 *
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

/*-------------------------------------- HIDDEN FUNCTIONS --------------------*/

// XXX XXX XXX
// Give the cuda packet a shm pointer instead of assuming it lies WITHIN the
// memory region itself. We can still place it in the memory region, but using a
// separate variable removes the assumption that it MUST be so.

/*
 * These functions are hidden so that you're forced to use the jump table
 * exported at the end of the file.
 */

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

static OPS_FN_PROTO(__CudaRegisterFatBinary)
{
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

	// Make the call, then add it to the cubin list.
	handle = __cudaRegisterFatBinary(cuda_cubin);
	pkt->ret_ex_val.handle = handle;
	cubins_add_cubin(cubin_list, cuda_cubin, handle);

	printd(DBG_DEBUG, "cudaRegFB handle=%p\n", pkt->ret_ex_val.handle);
	return 0;

fail:
	return -1;
}

static OPS_FN_PROTO(__CudaUnregisterFatBinary)
{
	void **handle = pkt->args[0].argdp;
	__cudaUnregisterFatBinary(handle);
	// FIXME Deallocate the fat binary data structures.
	pkt->ret_ex_val.err = cudaSuccess;
	printd(DBG_DEBUG, "unregister FB handle=%p\n", handle);
	return 0;
}

static OPS_FN_PROTO(__CudaRegisterFunction)
{
	int err, exit_errno;

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

static OPS_FN_PROTO(__CudaRegisterVar)
{
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

	// Make the call
	__cudaRegisterVar(uargs->fatCubinHandle, uargs->dom0HostAddr,
			uargs->deviceAddress, (const char *) uargs->deviceName,
			uargs->ext, uargs->size, uargs->constant, uargs->global);

	// store the state
	cubins_add_variable(cubin_list, uargs->fatCubinHandle, &uargs->link);
	pkt->ret_ex_val.err = cudaSuccess;
	printd(DBG_DEBUG, "_regVar: %p\n", uargs->hostVar);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(CudaSetDevice)
{
	int dev = pkt->args[0].argll;
	pkt->ret_ex_val.err = cudaSetDevice(dev);
	printd(DBG_DEBUG, "setDev %d\n", dev);
	return 0;
}

static OPS_FN_PROTO(CudaConfigureCall)
{
	dim3 gridDim = pkt->args[0].arg_dim;
	dim3 blockDim = pkt->args[1].arg_dim;
	size_t sharedMem = pkt->args[2].arr_argi[0];
	cudaStream_t stream = (cudaStream_t)pkt->args[3].argull;

	printd(DBG_DEBUG, "grid={%u,%u,%u} block={%u,%u,%u}"
			" shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

	pkt->ret_ex_val.err = 
		cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	return 0;
}

static OPS_FN_PROTO(CudaSetupArgument)
{
	const void *arg = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t size = pkt->args[1].arr_argi[0];
	size_t offset = pkt->args[1].arr_argi[1];
	pkt->ret_ex_val.err = cudaSetupArgument(arg, size, offset);
	printd(DBG_DEBUG, "setupArg arg=%p size=%lu offset=%lu ret=%u\n",
			arg, size, offset, pkt->ret_ex_val.err);
	return 0;
}

static OPS_FN_PROTO(CudaLaunch)
{
	int exit_errno;
	reg_func_args_t *func;

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubin_list = NULL;
	GET_CUBIN_VALIST(cubin_list, pkt);

	// FIXME We assume entry is just a memory pointer, not a string.
	// Printing the entry as a string will confuse your terminal
	const char *entry = (const char *)pkt->args[0].argull;

	// Locate the func structure; we assume func names are unique across cubins.
	bool found = false;
	cubins_for_each_cubin(cubin_list, fatcubin) {
		cubin_for_each_function(fatcubin, func) {
			if (func->hostFEaddr == entry) found = true;
			if (found) break;
		}
		if (found) break;
	}
	if (!found) {
		printd(DBG_ERROR, "launch cannot find entry func\n");
		exit_errno = -ENOENT;
		goto fail;
	}

	pkt->ret_ex_val.err = cudaLaunch(func->hostFun);
	printd(DBG_DEBUG, "launch(%p->%p)\n", entry, func->hostFun);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(CudaThreadExit)
{
	pkt->ret_ex_val.err = cudaThreadExit();
	printd(DBG_DEBUG, "thread exit\n");
	return 0;
}

static OPS_FN_PROTO(CudaThreadSynchronize)
{
	pkt->ret_ex_val.err = cudaThreadSynchronize();
	printd(DBG_DEBUG, "thread sync\n");
	return 0;
}

static OPS_FN_PROTO(CudaMalloc)
{
	// We are to write the value of devPtr to args[0].argull
	void *devPtr = NULL;
	size_t size = pkt->args[1].arr_argi[0];

	pkt->ret_ex_val.err = cudaMalloc(&devPtr, size);
	pkt->args[0].argull = (unsigned long long)devPtr;

	printd(DBG_DEBUG, "cudaMalloc devPtr=%p size=%lu ret:%u\n",
			devPtr, size, pkt->ret_ex_val.err);
	return 0;
}

static OPS_FN_PROTO(CudaFree)
{
	void *ptr = pkt->args[0].argp;
	pkt->ret_ex_val.err = cudaFree(ptr);
	printd(DBG_DEBUG, "free %p\n", ptr);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyH2D)
{
	void *dst = (void*)pkt->args[0].argull; // gpu ptr
	const void *src = (const void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	printd(DBG_DEBUG, "memcpyh2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2H)
{
	void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	void *src = (void*)pkt->args[1].argull; // gpu ptr
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	printd(DBG_DEBUG, "memcpyd2h dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyD2D)
{
	void *dst = (void*)pkt->args[0].argull;
	void *src = (void*)pkt->args[1].argull;
	size_t count = pkt->args[2].arr_argi[0];
	enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;

	pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
	printd(DBG_DEBUG, "memcpyd2d dst=%p src=%p count=%lu kind=%u\n",
			dst, src, count, kind);
	return 0;
}

static OPS_FN_PROTO(CudaMemcpyToSymbolH2D)
{
	int exit_errno;
	reg_var_args_t *var;

	const char *symbol = pkt->args[0].argcp;
	const void *src = (void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];

	struct cuda_fatcubin_info *fatcubin = NULL;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// Locate the var structure; symbols are unique across cubins. FIXME Here we
	// assume symbols are memory address; they may be strings instead.
	// See quote from A. Kerr in libci.c
	bool found = false;
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_variable(fatcubin, var) {
			if (var->hostVar == symbol) found = true;
			if (found) break;
		}
		if (found) break;
	}
	if (!found) {
		printd(DBG_ERROR, "memcpyFromSymb cannot find symbol\n");
		exit_errno = -ENOENT;
		goto fail;
	}

	pkt->ret_ex_val.err =
		cudaMemcpyToSymbol(var->dom0HostAddr, src, count, offset,
				cudaMemcpyHostToDevice);
	printd(DBG_DEBUG, "memcpyFromSymb %p\n", var->hostVar);
	return 0;

fail:
	return exit_errno;
}

static OPS_FN_PROTO(CudaMemcpyFromSymbolD2H)
{
	int exit_errno;
	reg_var_args_t *var;

	void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	const char *symbol = pkt->args[1].argcp;
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];

	struct cuda_fatcubin_info *fatcubin;
	struct fatcubins *cubins = NULL;
	GET_CUBIN_VALIST(cubins, pkt);

	// Locate the var structure; symbols are unique across cubins.
	// See quote from A. Kerr in libci.c
	bool found = false;
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_variable(fatcubin, var) {
			// FIXME we assume symbols are memory addresses here. The
			// CUDA Runtime API says they may also be strings, if the
			// app desires.
			if (var->hostVar == symbol) found = true;
			if (found) break;
		}
		if (found) break;
	}
	if (!found) {
		printd(DBG_ERROR, "memcpyFromSymb cannot find symbol\n");
		exit_errno = -ENOENT;
		goto fail;
	}
	pkt->ret_ex_val.err =
		cudaMemcpyFromSymbol(dst, var->dom0HostAddr, count, offset,
				cudaMemcpyDeviceToHost);
	printd(DBG_DEBUG, "memcpyFromSymb %p\n", var->hostVar);
	return 0;

fail:
	return exit_errno;
}

const struct cuda_ops exec_ops =
{
	// Functions which take only a cuda_packet*
	.configureCall = CudaConfigureCall,
	.free = CudaFree,
	.malloc = CudaMalloc,
	.memcpyD2D = CudaMemcpyD2D,
	.memcpyD2H = CudaMemcpyD2H,
	.memcpyH2D = CudaMemcpyH2D,
	.setDevice = CudaSetDevice,
	.setupArgument = CudaSetupArgument,
	.threadExit = CudaThreadExit,
	.threadSynchronize = CudaThreadSynchronize,
	.unregisterFatBinary = __CudaUnregisterFatBinary,

	// Functions which take a cuda_packet* and fatcubins*
	.launch = CudaLaunch,
	.memcpyFromSymbolD2H = CudaMemcpyFromSymbolD2H,
	.memcpyToSymbolH2D = CudaMemcpyToSymbolH2D,
	.registerFatBinary = __CudaRegisterFatBinary,
	.registerFunction = __CudaRegisterFunction,
	.registerVar = __CudaRegisterVar,
};

/*--------------------------- ^^ CODE TO MERGE UP ^^ ---------------------*/

#if 0
case CUDA_MEMCPY_FROM_SYMBOL_D2D:
{
	struct cuda_fatcubin_info *fatcubin;
	int found = 0; // 0 if var not found, 1 otherwise
	reg_var_args_t *var;
	void *dst = pkt->args[0].argull;
	const char *symbol = pkt->args[1].argcp;
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];
	// Locate the var structure; symbols are unique across cubins.
	// See quote from A. Kerr in libci.c
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_variable(fatcubin, var) {
			// FIXME we assume symbols are memory addresses here. The
			// CUDA Runtime API says they may also be strings, if the
			// app desires.
			if (var->hostVar == symbol) found = 1;
			if (found) break;
		}
		if (found) break;
	}
	if (unlikely(!found)) {
		printd(DBG_ERROR, "memcpyFromSymb cannot find symbol\n");
		fail = 1;
		break;
	}
	printd(DBG_DEBUG, "memcpyFromSymb %p\n", var->hostVar);
	pkt->ret_ex_val.err =
		cudaMemcpyFromSymbol(dst, var->dom0HostAddr, count, offset,
				cudaMemcpyDeviceToDevice);
}
break;
#endif

#if 0
	if (pkt->method_id == __CUDA_REGISTER_FAT_BINARY &&
			pkt->ret_ex_val.handle == NULL) {
		printd(DBG_ERROR, "__cudaRegFatBin returned NULL handle\n");
	} else if (pkt->method_id != __CUDA_REGISTER_FAT_BINARY &&
			pkt->ret_ex_val.err != cudaSuccess) {
		printd(DBG_ERROR, "method %u returned not cudaSuccess: %u\n",
				pkt->method_id, pkt->ret_ex_val.err);
	}
#endif

#if 0
case CUDA_MEMCPY_TO_SYMBOL_D2D:
{
	struct cuda_fatcubin_info *fatcubin;
	reg_var_args_t *var;
	int found = 0; // 0 if var not found, 1 otherwise
	const char *symbol = pkt->args[0].argcp;
	const void *src = pkt->args[1].argull;
	size_t count = pkt->args[2].arr_argi[0];
	size_t offset = pkt->args[2].arr_argi[1];
	// Locate the var structure; symbols are unique across cubins.
	// See quote from A. Kerr in libci.c
	cubins_for_each_cubin(cubins, fatcubin) {
		cubin_for_each_variable(fatcubin, var) {
			// FIXME we assume symbols are memory addresses here. The
			// CUDA Runtime API says they may also be strings, if the
			// app desires.
			if (var->hostVar == symbol) found = 1;
			if (found) break;
		}
		if (found) break;
	}
	if (unlikely(!found)) {
		printd(DBG_ERROR, "memcpyFromSymb cannot find symbol\n");
		fail = 1;
		break;
	}
	printd(DBG_DEBUG, "memcpyFromSymb %p\n", var->hostVar);
	pkt->ret_ex_val.err =
		cudaMemcpyToSymbol(var->dom0HostAddr, src, count, offset,
				cudaMemcpyDeviceToDevice);
}
break;
#endif

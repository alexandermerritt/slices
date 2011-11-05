/**
 * @file localsink.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-27
 *
 * @brief TODO
 */

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <unistd.h>

#include <assembly.h>
#include <common/libregistration.h>
#include <debug.h>
#include <fatcubininfo.h>
#include <libciutils.h>
#include <method_id.h>
#include <packetheader.h>
#include <util/x86_system.h>

#include "sinks.h"

/* Not defined in any header? */
extern void** __cudaRegisterFatBinary(void*);
extern void __cudaUnregisterFatBinary(void**);
extern void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global);

//#undef printd
//#define printd(level, fmt, args...)

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct fatcubins *cubins;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/


/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

/**
 * Entry point for the local sink process. It is forked from runtime when a new
 * application process registers. This function should never exit.
 *
 * This function shall NOT call any functions that request/destroy
 * assemblies, nor those that modify registration state (or any library state
 * for that matter). That is under the control of the runtime process when it
 * detects new applications joining or leaving. This state will exist from the
 * fork, but modifying it will not enable it to be visible to the runtime
 * process. TODO Unless all the state is stored in a shared memory region that
 * can be accessed across forks().
 *
 * Since we were forked, we are single-threaded. fork() forks only the invoking
 * thread, not all threads in a process. Thus if the runtime was created as an
 * assembly main node it would spawn an assembly RPC thread; this will not exist
 * within us.  However, all file descriptors and memory pages are identical.
 *
 * Forking this may be dangerous from within the runtime process. Please read:
 * http://www.linuxprogrammingblog.com/threads-and-fork-think-twice-before-using-them
 * Issues that may arise here include locks that can never be acquired in the
 * assembly code: fork() may have been called while a thread was in the middle
 * of a locked critical section. Since fork does not duplicate that thread, it
 * will not exist to unlock anything. If anything suspicious does happen, we can
 * try spawning a thread to handle this function instead of a new process
 * altogether.
 */
void localsink(asmid_t asmid, regid_t regid)
{
	int err;
	void *shm = NULL;
	volatile struct cuda_packet *cpkt_shm = NULL;

	printd(DBG_INFO, "localsink spawned, pid=%d asm=%lu regid=%d\n",
			getpid(), asmid, regid);

	// Allocate cubin set.
	cubins = calloc(1, sizeof(*cubins));
	if (!cubins) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		return;
	}
	cubins_init(cubins);

	shm = reg_be_get_shm(regid);

	// Process the shm for cuda packets.
	// TODO Once the lib uses multiple threads, we will need to consider state
	// that divides the shm region into per-thread regions (OR, do what Vishakha
	// did in the driver within VMs: have a tree-based memory allocator). We can
	// start with one thread looking at all queues. Then migrate to spawning a
	// set of worker threads, one per queue. When linked with a scheduling
	// library, we can add specific threads to be scheduled, too.

	cpkt_shm = (struct cuda_packet *)shm;
	while (1) {

		// wait for next packet
		printd(DBG_DEBUG, "waiting for next packet...\n");
		rmb();
		while (!(cpkt_shm->flags & CUDA_PKT_REQUEST))
			rmb();

		// Tell assembly to execute the packet for us
		// TODO Use a thread pool to associate with each thread in the
		// application.
		err = assembly_rpc(asmid, 0, cpkt_shm); // FIXME don't hardcode vgpu 0
		if (err < 0) {
			printd(DBG_ERROR, "assembly_rpc returned error\n");
			// what else to do? this is supposed to be an error of the assembly
			// module, not an error of the call specifically
			assert(0);
		}

		// Don't forget! Indicate packet is executed and on return path.
		cpkt_shm->flags = CUDA_PKT_RESPONSE;
		wmb();
	}

	assert(0);
}

/**
 * Execute a cuda packet. The assembly layer calls into this if it receives a
 * call destined for a local physical GPU.
 *
 * It is assumed the address pointed to by the argument pkt is located in a
 * shared memory region. Arguments within the packet, if for a function
 * prototype with pointer arguments, contain values offset from the address of
 * pkt at which the serialized arguments can be found or should be stored.
 *
 * FIXME What can be done to change this, is to have the packet contain a
 * variable pointing to an address region that the argument offsets then index
 * into, instead of assuming the packet itself is inside the shared region.
 */
int nv_exec_pkt(volatile struct cuda_packet *pkt)
{
	int err;
	int fail = 0; // 1 to indicate failure within the switch
	switch (pkt->method_id) {

		/*
		 * Functions handled by the assembly layer above.
		 */

		case CUDA_GET_DEVICE:
		case CUDA_GET_DEVICE_COUNT:
		case CUDA_GET_DEVICE_PROPERTIES:
		case CUDA_DRIVER_GET_VERSION:
		case CUDA_RUNTIME_GET_VERSION:
		{
			printd(DBG_ERROR, "Error: packet requesting assembly information\n");
			fail = 1;
		}
		break;

		/*
		 * Hidden functions for registration.
		 */

		case __CUDA_REGISTER_FAT_BINARY:
		{
			void *handle;
			void *cubin_shm = ((void*)pkt + pkt->args[0].argull);
			__cudaFatCudaBinary *cuda_cubin =
				calloc(1, sizeof(__cudaFatCudaBinary));
			if (!cuda_cubin) {
				printd(DBG_ERROR, "out of memory\n");
				fprintf(stderr, "out of memory\n");
				fail = 1;
				break;
			}
			err = unpackFatBinary(cuda_cubin, cubin_shm);
			if (err < 0) {
				printd(DBG_ERROR, "error unpacking fat cubin\n");
				fail = 1;
				break;
			}
			handle = __cudaRegisterFatBinary(cuda_cubin);
			pkt->ret_ex_val.handle = handle;
			cubins_add_cubin(cubins, cuda_cubin, handle);
			printd(DBG_DEBUG, "cudaRegFB handle=%p\n", pkt->ret_ex_val.handle);
		}
		break;

		case __CUDA_UNREGISTER_FAT_BINARY:
		{
			void **handle = pkt->args[0].argdp;
			__cudaUnregisterFatBinary(handle);
			// FIXME Deallocate the fat binary data structures.
			pkt->ret_ex_val.err = cudaSuccess;
		}
		break;

		case __CUDA_REGISTER_FUNCTION:
		{
			// unpack the serialized arguments from shared mem
			reg_func_args_t *pargs =  // packed
				((void*)pkt + pkt->args[0].argull);
			reg_func_args_t *uargs = // unpacked
				calloc(1, sizeof(reg_func_args_t));
			if (!uargs) {
				printd(DBG_ERROR, "out of memory\n");
				fprintf(stderr, "out of memory\n");
				fail = 1;
				break;
			}
			// FIXME don't use char*
			err = unpackRegFuncArgs(uargs, (char *)pargs);
			if (err < 0) {
				printd(DBG_ERROR, "error unpacking regfunc args\n");
				fail = 1;
				break;
			}
			__cudaRegisterFunction(uargs->fatCubinHandle,
					(const char *)uargs->hostFun, uargs->deviceFun,
					(const char *)uargs->deviceName, uargs->thread_limit,
					uargs->tid, uargs->bid, uargs->bDim, uargs->gDim,
					uargs->wSize);
			// store the state
			cubins_add_function(cubins, uargs->fatCubinHandle, &uargs->link);
			pkt->ret_ex_val.err = cudaSuccess;
			printd(DBG_DEBUG, "regFunc handle=%p\n", uargs->fatCubinHandle);
		}
		break;

		case __CUDA_REGISTER_VARIABLE:
		{
			// unpack the serialized arguments from shared mem
			reg_var_args_t *pargs = // packed
				((void*)pkt + pkt->args[0].argull);
			reg_var_args_t *uargs = // unpacked
				calloc(1, sizeof(reg_var_args_t));
			if (!uargs) {
				printd(DBG_ERROR, "out of memory\n");
				fprintf(stderr, "out of memory\n");
				fail = 1;
				break;
			}
			err = unpackRegVar(uargs, (char *)pargs);
			if (err < 0) {
				printd(DBG_ERROR, "error unpacking regvar args\n");
				fail = 1;
				break;
			}
			printd(DBG_DEBUG, "_regVar: %p\n", uargs->hostVar);
			__cudaRegisterVar(uargs->fatCubinHandle, uargs->dom0HostAddr,
					uargs->deviceAddress, (const char *) uargs->deviceName,
					uargs->ext, uargs->size, uargs->constant, uargs->global);
			// store the state
			cubins_add_variable(cubins, uargs->fatCubinHandle, &uargs->link);
			pkt->ret_ex_val.err = cudaSuccess;
		}
		break;

		/*
		 * Public CUDA runtime functions.
		 */

		case CUDA_SET_DEVICE:
		{
			int dev = pkt->args[0].argll;
			pkt->ret_ex_val.err = cudaSetDevice(dev);
			printd(DBG_DEBUG, "setDev %d\n", dev);
		}
		break;

		case CUDA_CONFIGURE_CALL:
		{
			dim3 gridDim = pkt->args[0].arg_dim;
			dim3 blockDim = pkt->args[1].arg_dim;
			size_t sharedMem = pkt->args[2].arr_argi[0];
			cudaStream_t stream = (void*)pkt->args[3].argull;

			printd(DBG_DEBUG, "grid={%d,%d,%d} block={%d,%d,%d}"
					" shmem=%lu strm=%p\n",
					gridDim.x, gridDim.y, gridDim.z,
					blockDim.x, blockDim.y, blockDim.z,
					sharedMem, stream);

			pkt->ret_ex_val.err = 
				cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
		}
		break;

		case CUDA_SETUP_ARGUMENT:
		{
			const void *arg = ((void*)pkt + pkt->args[0].argull);
			size_t size = pkt->args[1].arr_argi[0];
			size_t offset = pkt->args[1].arr_argi[1];
			pkt->ret_ex_val.err = cudaSetupArgument(arg, size, offset);
			printd(DBG_DEBUG, "setupArg arg=%p size=%lu offset=%lu ret=%d\n",
					arg, size, offset, pkt->ret_ex_val.err);
		}
		break;

		case CUDA_LAUNCH:
		{
			// FIXME We assume entry is just a memory pointer, not a string.
			// Printing the entry as a string will confuse your terminal
			const char *entry = (void*)pkt->args[0].argull;
			struct cuda_fatcubin_info *fatcubin;
			reg_func_args_t *func;
			int found = 0; // 0 if func not found, 1 otherwise
			// Locate the func structure; we assume func names are unique across
			// cubins.
			cubins_for_each_cubin(cubins, fatcubin) {
				cubin_for_each_function(fatcubin, func) {
					if (func->hostFEaddr == entry) found = 1;
					if (found) break;
				}
				if (found) break;
			}
			if (unlikely(!found)) {
				printd(DBG_ERROR, "launch cannot find entry func\n");
				fail = 1;
				break;
			}
			printd(DBG_DEBUG, "launch(%p->%p)\n", entry, func->hostFun);
			pkt->ret_ex_val.err = cudaLaunch(func->hostFun);
		}
		break;

		case CUDA_THREAD_EXIT:
		{
			pkt->ret_ex_val.err = cudaThreadExit();
			printd(DBG_DEBUG, "thread exit\n");
		}
		break;

		case CUDA_THREAD_SYNCHRONIZE:
		{
			pkt->ret_ex_val.err = cudaThreadSynchronize();
			printd(DBG_DEBUG, "thread sync\n");
		}
		break;

		case CUDA_MALLOC:
		{
			// We are to write the value of devPtr to args[0].argull
			void *devPtr = NULL;
			size_t size = pkt->args[1].arr_argi[0];
			pkt->ret_ex_val.err = cudaMalloc(&devPtr, size);
			printd(DBG_DEBUG, "cudaMalloc devPtr=%p size=%lu ret:%d\n",
					devPtr, size, pkt->ret_ex_val.err);
			pkt->args[0].argull = (unsigned long long)devPtr;
		}
		break;

		case CUDA_FREE:
		{
			void *ptr = pkt->args[0].argp;
			pkt->ret_ex_val.err = cudaFree(ptr);
			printd(DBG_DEBUG, "free %p\n", ptr);
		}
		break;

		case CUDA_MEMCPY_H2D:
		{
			void *dst = (void*)pkt->args[0].argull; // gpu ptr
			const void *src = ((void*)pkt + pkt->args[1].argull);
			size_t count = pkt->args[2].arr_argi[0];
			enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
			pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
			printd(DBG_DEBUG, "memcpyh2d dst=%p src=%p count=%lu kind=%d\n",
					dst, src, count, kind);
		}
		break;

		case CUDA_MEMCPY_D2H:
		{
			void *dst = ((void*)pkt + pkt->args[0].argull);
			void *src = (void*)pkt->args[1].argull; // gpu ptr
			size_t count = pkt->args[2].arr_argi[0];
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
			pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
			printd(DBG_DEBUG, "memcpyd2h dst=%p src=%p count=%lu kind=%d\n",
					dst, src, count, kind);
		}
		break;

		case CUDA_MEMCPY_D2D:
		{
			void *dst = (void*)pkt->args[0].argull;
			void *src = (void*)pkt->args[1].argull;
			size_t count = pkt->args[2].arr_argi[0];
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
			pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
			printd(DBG_DEBUG, "memcpyd2d dst=%p src=%p count=%lu kind=%d\n",
					dst, src, count, kind);
		}
		break;

		case CUDA_MEMCPY_TO_SYMBOL_H2D:
		{
			struct cuda_fatcubin_info *fatcubin;
			reg_var_args_t *var;
			int found = 0; // 0 if var not found, 1 otherwise
			const char *symbol = pkt->args[0].argcp;
			const void *src = ((void*)pkt + pkt->args[1].argull);
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
									cudaMemcpyHostToDevice);
		}
		break;

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

		case CUDA_MEMCPY_FROM_SYMBOL_D2H:
		{
			struct cuda_fatcubin_info *fatcubin;
			int found = 0; // 0 if var not found, 1 otherwise
			reg_var_args_t *var;
			void *dst = ((void*)pkt + pkt->args[0].argull);
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
						cudaMemcpyDeviceToHost);
		}
		break;

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

		/*
		 * All other functions are unhandled.
		 */

		default:
		{
			printd(DBG_ERROR, "Unhandled packet, method %d\n", pkt->method_id);
			pkt->ret_ex_val.err = (!cudaSuccess);
			fail = 1;
		}
		break;
	}
	if (fail)
		return -1;
	if (pkt->method_id == __CUDA_REGISTER_FAT_BINARY &&
			pkt->ret_ex_val.handle == NULL) {
		printd(DBG_ERROR, "__cudaRegFatBin returned NULL handle\n");
	} else if (pkt->method_id != __CUDA_REGISTER_FAT_BINARY &&
			pkt->ret_ex_val.err != cudaSuccess) {
		printd(DBG_ERROR, "method %d returned not cudaSuccess: %d\n",
				pkt->method_id, pkt->ret_ex_val.err);
	}
	return 0;
}

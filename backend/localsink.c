/**
 * @file localsink.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-27
 *
 * @brief TODO
 */

// System includes
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// Project includes
#include <assembly.h>
#include <cuda_hidden.h>
#include <debug.h>
#include <fatcubininfo.h>
#include <libciutils.h>
#include <method_id.h>
#include <packetheader.h>
#include <shmgrp.h>
#include <util/x86_system.h>

// Directory-immediate includes
#include "sinks.h"

//#undef printd
//#define printd(level, fmt, args...)

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct fatcubins *cubins;

// FIXME FIXME We are assuming SINGLE-THREADED CUDA applications as a first
// step, thus one region and one address.
struct shmgrp_region region;
void *shm;

bool process_rpc = false; //! Barrier and stop flag for main processing loop
bool loop_exited = false; //! Indication that the main loop has exited

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

// We assume a new region request maps 1:1 to a new thread (using the CUDA API)
// having spawned in the CUDA application.
// TODO Spawn a thread for each new shm region, and cancel it on remove.
static void region_request(shm_event e, pid_t memb_pid, shmgrp_region_id id)
{
	int err;
	switch (e) {
		case SHM_CREATE_REGION:
		{
			err = shmgrp_member_region(ASSEMBLY_SHMGRP_KEY, memb_pid, id, &region);
			if (err < 0) {
				printd(DBG_ERROR, "Could not access region %d details"
						" for process %d\n", id, memb_pid);
				break;
			}
			shm = region.addr;
			printd(DBG_DEBUG, "Process %d created a shm region:"
					" id=%d @%p bytes=%lu\n",
					memb_pid, id, region.addr, region.size);
			fflush(stdout);
			process_rpc = true; // Set this last, after shm has been set
		}
		break;
		case SHM_REMOVE_REGION:
		{
			process_rpc = false;
			err = shmgrp_member_region(ASSEMBLY_SHMGRP_KEY, memb_pid, id, &region);
			if (err < 0) {
				printd(DBG_ERROR, "Could not access region %d details"
						" for process %d\n", id, memb_pid);
				break;
			}
			printd(DBG_DEBUG, "Process %d destroyed an shm region:"
					" id=%d @%p bytes=%lu\n",
					memb_pid, id, region.addr, region.size);
			fflush(stdout);
			// wait for the loop to see our change to process_rpc before
			// continuing, to avoid it reading the shm after it has been
			// unmapped
			while (!loop_exited)
				;
		}
		break;
	}
}

#ifdef LOCALSINK_USE_THREAD

struct thread_localsink_arg
{
	asmid_t asmid;
	pid_t pid;
};

static void * thread_localsink(void * arg)
{
	struct thread_localsink_arg *args =
		(struct thread_localsink_arg *) arg;
	localsink(args->asmid, args->pid);
	pthread_exit(NULL);
}

#else			/* !LOCALSINK_USE_THREAD */

static void sigterm_handler(int sig)
{
	; // Do nothing, just prevent it from killing us
}

#endif			/* LOCALSINK_USE_THREAD */

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

/**
 * The main() function for forked sink processes. It must not be allowed to
 * return within the child process. Once it completes its work, it waits on a
 * termination signal from the runtime coordinator before exiting. That signal
 * is sent when the CUDA process tells the runtime it wants to leave the
 * assembly runtime.
 *
 * The parent process will simply leave this function after invoking fork.
 */
int fork_localsink(asmid_t asmid, pid_t memb_pid, pid_t *child_pid)
{
	int err;

#ifdef LOCALSINK_USE_THREAD

	pthread_t sink_tid;

	struct thread_localsink_arg *arg = malloc(sizeof(*arg));
	if (!arg)
		goto fail;
	arg->asmid = asmid;
	arg->pid = memb_pid;

	err = pthread_create(&sink_tid, NULL, thread_localsink, arg);
	if (err < 0) {
		printd(DBG_ERROR, "Error creating pthread for localsink\n");
		goto fail;
	}
	printd(DBG_INFO, "Spawned thread for localsink\n");

	return 0;

fail:
	return -1;

#else			/* !LOCALSINK_USE_THREAD */

	sigset_t mask;
	struct sigaction action;

	printd(DBG_INFO, "Forking localsink\n");

	*child_pid = fork();

	// Fork error.
	if (*child_pid < 0) {
		printd(DBG_ERROR, "fork() failed, pid=%d\n", *child_pid);
		goto fail;
	}

	// Parent does nothing more.
	if (*child_pid > 0) {
		return 0;
	}

	/*
	 * We're the child process.
	 */

	// Install a handler for the termination signal.
	memset(&action, 0, sizeof(action));
	action.sa_handler = sigterm_handler;
	sigemptyset(&action.sa_mask);
	err = sigaction(SINK_TERM_SIG, &action, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Error installing signal handler for SIGTERM.\n");
	}

	localsink(asmid, memb_pid); // Process all RPC calls.

	printd(DBG_INFO, "Waiting for runtime to signal us\n");

	// Wait for the runtime to terminate us.
	sigdelset(&mask, SINK_TERM_SIG);
	sigsuspend(&mask);

	printd(DBG_INFO, "Destroying association with member %d\n", memb_pid);
	err = shmgrp_destroy_member(ASSEMBLY_SHMGRP_KEY, memb_pid);
	if (err < 0) {
		printd(DBG_ERROR, "Error destroying member %d state: %s\n",
				memb_pid, strerror(-(err)));
	}

	exit(0);

fail:
	exit(-1);
#endif			/* LOCALSINK_USE_THREAD */
}

/**
 * Entry point for the local sink process. It is forked from runtime when a new
 * application process registers. Right now it directly executes the RPCs, but
 * in future code TODO it should only register with the CUDA app it was forked
 * to handle (establish_member) then wait on a signal or something. Each time
 * the shm_callback is invoked, that function should spawn a thread to handle
 * the region created (as a CUDA thread will be pushing RPCs through it).
 *
 * This file shall NOT call any functions that request/destroy assemblies, nor
 * those that modify shmgrp state (or any library state for that matter) EXCEPT
 * to establish/destroy members. Library and other state will exist from the
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
void localsink(asmid_t asmid, pid_t pid)
{
	int err;
	volatile struct cuda_packet *cpkt_shm = NULL;

	printd(DBG_INFO, "localsink spawned, asmid=%lu pid=%d,"
			" attaching to cuda pid=%d\n", asmid, getpid(), pid);

	// Tell shmgrp we are assuming responsibility for this process.
	//
	// developer note: establish_member will set notification on the underlying
	// MQs within the library; this creates threads.
	err = shmgrp_establish_member(ASSEMBLY_SHMGRP_KEY, pid, region_request);
	if (err < 0) {
		printd(DBG_ERROR, "Could not establish member %d\n", pid);
		exit(-1);
	}

	// Allocate cubin set.
	cubins = calloc(1, sizeof(*cubins));
	if (!cubins) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		return;
	}
	cubins_init(cubins);

	printd(DBG_INFO, "Spinning until application creates a shm region\n");

	// FIXME FIXME HACK
	// The shm_callback will modify this variable. If a region is registered, it
	// will enable it, and vice versa. We assume single-threaded cuda
	// applications, performing one single mkreg and rmreg.
	while (!process_rpc)
		;

	printd(DBG_INFO, "Now proceeding to process CUDA RPCs\n");

	// Process CUDA RPC until the CUDA application unregisters its memory
	// region.
	cpkt_shm = (struct cuda_packet *)shm;
	while (process_rpc) {

		// Check packet flag. If not ready, check loop flag. Spin.
		rmb();
		if (!(cpkt_shm->flags & CUDA_PKT_REQUEST)) {
			rmb();
			continue;
		}

		// Tell assembly to execute the packet for us
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
	loop_exited = true;

	// FIXME Destroy cubins.
}

/**
 * Execute a cuda packet. The assembly layer calls into this if it receives a
 * call destined for a local physical GPU.
 *
 * FIXME Move the execution of a packet to a separate file/interface.
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
			void *cubin_shm = (void*)((uintptr_t)pkt + pkt->args[0].argull);
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
			printd(DBG_DEBUG, "unregister FB handle=%p\n", handle);
		}
		break;

		case __CUDA_REGISTER_FUNCTION:
		{
			// unpack the serialized arguments from shared mem
			reg_func_args_t *pargs =  // packed
				(reg_func_args_t*)((uintptr_t)pkt + pkt->args[0].argull);
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
				(reg_var_args_t*)((uintptr_t)pkt + pkt->args[0].argull);
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
			cudaStream_t stream = (cudaStream_t)pkt->args[3].argull;

			printd(DBG_DEBUG, "grid={%u,%u,%u} block={%u,%u,%u}"
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
			const void *arg = (void*)((uintptr_t)pkt + pkt->args[0].argull);
			size_t size = pkt->args[1].arr_argi[0];
			size_t offset = pkt->args[1].arr_argi[1];
			pkt->ret_ex_val.err = cudaSetupArgument(arg, size, offset);
			printd(DBG_DEBUG, "setupArg arg=%p size=%lu offset=%lu ret=%u\n",
					arg, size, offset, pkt->ret_ex_val.err);
		}
		break;

		case CUDA_LAUNCH:
		{
			// FIXME We assume entry is just a memory pointer, not a string.
			// Printing the entry as a string will confuse your terminal
			const char *entry = (const char *)pkt->args[0].argull;
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
			printd(DBG_DEBUG, "cudaMalloc devPtr=%p size=%lu ret:%u\n",
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
			const void *src = (const void*)((uintptr_t)pkt + pkt->args[1].argull);
			size_t count = pkt->args[2].arr_argi[0];
			enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;
			pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
			printd(DBG_DEBUG, "memcpyh2d dst=%p src=%p count=%lu kind=%u\n",
					dst, src, count, kind);
		}
		break;

		case CUDA_MEMCPY_D2H:
		{
			void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
			void *src = (void*)pkt->args[1].argull; // gpu ptr
			size_t count = pkt->args[2].arr_argi[0];
			enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
			pkt->ret_ex_val.err = cudaMemcpy(dst, src, count, kind);
			printd(DBG_DEBUG, "memcpyd2h dst=%p src=%p count=%lu kind=%u\n",
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
			printd(DBG_DEBUG, "memcpyd2d dst=%p src=%p count=%lu kind=%u\n",
					dst, src, count, kind);
		}
		break;

		case CUDA_MEMCPY_TO_SYMBOL_H2D:
		{
			struct cuda_fatcubin_info *fatcubin;
			reg_var_args_t *var;
			int found = 0; // 0 if var not found, 1 otherwise
			const char *symbol = pkt->args[0].argcp;
			const void *src = (void*)((uintptr_t)pkt + pkt->args[1].argull);
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
			void *dst = (void*)((uintptr_t)pkt + pkt->args[0].argull);
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
			printd(DBG_ERROR, "Unhandled packet, method %u\n", pkt->method_id);
			pkt->ret_ex_val.err = cudaErrorUnknown;
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
		printd(DBG_ERROR, "method %u returned not cudaSuccess: %u\n",
				pkt->method_id, pkt->ret_ex_val.err);
	}
	return 0;
}

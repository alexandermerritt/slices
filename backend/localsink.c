/**
 * @file localsink.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-27
 *
 * @brief TODO
 */

#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <util/x86_system.h>

#include <assembly.h>
#include <common/libregistration.h>
#include <debug.h>
#include <method_id.h>
#include <packetheader.h>

#include "sinks.h"

/*-------------------------------------- INTERNAL STATE ----------------------*/


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
	int num_gpus, err;
	void *shm = NULL;
	size_t shm_sz;
	volatile struct cuda_packet *cpkt_shm = NULL;

	printd(DBG_INFO, "localsink spawned, pid=%d asm=%lu regid=%d\n",
			getpid(), asmid, regid);

	num_gpus = assembly_num_vgpus(asmid);
	shm = reg_be_get_shm(regid);
	shm_sz = reg_be_get_shm_size(regid);

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

#if 0
		// tell assembly to execute the packet for us
		err = assembly_rpc(asmid, 0, cpkt_shm); // FIXME don't hardcode vgpu 0
		if (err < 0) {
			printd(DBG_ERROR, "error executing assembly_rpc\n");
			// what else to do?
			assert(0);
		}
#endif

		// FIXME Remove execution of packet from here.
		switch (cpkt_shm->method_id) {

			case CUDA_GET_DEVICE_COUNT:
				{
					int *devs = (int*)((void*)cpkt_shm + cpkt_shm->args[0].argull);
					printd(DBG_INFO, "executing cudaGetDeviceCount(%p)\n", devs);
					cudaGetDeviceCount(devs);
					printd(DBG_DEBUG, "cudaGetDevieCount returned %d\n", *devs);
				}
				break;

			case CUDA_GET_DEVICE_PROPERTIES:
				{
					struct cudaDeviceProp *prop_shm =
						(struct cudaDeviceProp*)((void*)cpkt_shm +
								cpkt_shm->args[0].argull);
					int dev = cpkt_shm->args[1].argll;
					printd(DBG_INFO, "executing cudaGetDeviceProperties(%p, %d)\n",
							prop_shm, dev);
					cudaGetDeviceProperties(prop_shm, dev);
					printd(DBG_DEBUG, "cudaGetDevieCount returned\n");
				}
				break;

			default:
				printd(DBG_ERROR, "unhandled method: %d\n", cpkt_shm->method_id);
				assert(0);
				break;
		}

		// Don't forget! Indicate packet is executed and on return path.
		cpkt_shm->flags = CUDA_PKT_RESPONSE;
		wmb();
	}

	assert(0);
	__builtin_unreachable();
}

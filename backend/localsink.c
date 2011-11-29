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

/*-------------------------------------- EXTERNAL VARIABLES ------------------*/

// POSIX environment variable list, man 7 environ
extern char **environ;

/*-------------------------------------- INTERNAL STATE ----------------------*/

// FIXME FIXME We are assuming SINGLE-THREADED CUDA applications as a first
// step, thus one region and one address.
static struct shmgrp_region region;
static void *shm;
static pid_t pid; // of app we're working with

static bool process_rpc = false; //! Barrier and stop flag for main processing loop
static bool loop_exited = false; //! Indication that the main loop has exited

static asmid_t asmid;
static assembly_key_uuid uuid; // key of assembly metadata we import

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
			pid = memb_pid;
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

static void
sigterm_handler(int sig)
{
	; // Do nothing, just prevent it from killing us
}

// forward declare the hidden shmgrp functions
extern int __shmgrp_insert_group(const char *);
extern int __shmgrp_dump_group(const char *);

static int
setup(void)
{
	int err;

	BUG(!environ);

	// Get the UUID to import the assembly with.
	char *uuid_env = getenv(SINK_ASSM_EXPORT_KEY_ENV);
	if (!uuid_env) {
		printd(DBG_ERROR, "Assembly env key not defined.\n");
		return -1;
	}
	err = uuid_parse(uuid_env, uuid);
	if (err < 0) {
		printd(DBG_ERROR, "Invalid key {%s}\n", uuid_env);
		return -1;
	}
	// Get the application PID we are assuming responsibility for.
	char *pid_env = getenv(SINK_SHMGRP_PID);
	if (!pid_env) {
		printd(DBG_ERROR, "Group member PID not defined.\n");
		return -1;
	}
	pid = atoi(pid_env);

	// Tell the assembly module we only import & map assemblies.
	err = assembly_runtime_init(NODE_TYPE_MAPPER, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize assembly runtime.\n");
		return -1;
	}
	// Import the assembly given to us.
	err = assembly_import(&asmid, uuid);
	if (err < 0) {
		printd(DBG_ERROR, "Could not import assembly, key={%s}\n", uuid_env);
		return -1;
	}
	BUG(asmid == INVALID_ASSEMBLY_ID);
	// Set up the function table and establish remote paths, if needed
	err = assembly_map(asmid);
	if (err < 0) {
		printd(DBG_ERROR, "Could not map assembly %lu\n", asmid);
		return -1;
	}

	// Initialize shmgrp state
	err = shmgrp_init();
	if (err < 0) {
		printd(DBG_ERROR, "Could not init shmgrp\n");
		return -1;
	}
	// Re-insert shmgrp state necessary to establish our member (application).
	err = __shmgrp_insert_group(ASSEMBLY_SHMGRP_KEY);
	if (err < 0) {
		printd(DBG_ERROR, "Could not insert group\n");
		return -1;
	}
	// Tell shmgrp we are assuming responsibility for this process. Do this
	// last, as it turns on async notification of memory region creations from
	// the application.
	err = shmgrp_establish_member(ASSEMBLY_SHMGRP_KEY, pid, region_request);
	if (err < 0) {
		printd(DBG_ERROR, "Could not establish member %d\n", pid);
		return -1;
	}

	// Install a handler for our term sig so it doesn't kill us
	struct sigaction action;
	memset(&action, 0, sizeof(action));
	action.sa_handler = sigterm_handler;
	sigemptyset(&action.sa_mask);
	err = sigaction(SINK_TERM_SIG, &action, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Could not install sig handler.\n");
		return -1;
	}

	return 0;
}

static void
wait_for_termination(void)
{
	sigset_t mask;
	sigemptyset(&mask);
	sigsuspend(&mask);
}

// Tell the compiler all calls to teardown never return.
static void teardown(void) __attribute__((noreturn));

static void
teardown(void)
{
	int err;
	// Make associated calls in the reverse order they appear in within setup()
	err = shmgrp_destroy_member(ASSEMBLY_SHMGRP_KEY, pid);
	if (err < 0)
		printd(DBG_ERROR, "Could not destroy member %d\n", pid);
	err = __shmgrp_dump_group(ASSEMBLY_SHMGRP_KEY);
	err = assembly_runtime_shutdown();
	err = shmgrp_tini();
	_exit(0);
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

/**
 * Entry point for the local sink process. It is forked/exec'd from runtime when
 * a new application process registers.
 *
 * This file shall NOT call any functions that request/destroy assemblies, nor
 * those that modify shmgrp state (or any library state for that matter) EXCEPT
 * to establish/destroy members. From the shmgrp perspective, it is a leader, of
 * the same group the runtime is, but will receive no notifications of joining
 * members (via that double-underscore function call, instead of actually
 * opening a new group altogether).
 *
 * We expect argc to be zero, which tells us this was exec'd and not run from
 * the command line (it's normally > 0).
 */
int main(int argc, char *argv[])
{
	int err;
	struct cuda_packet *cpkt_shm = NULL;

	// Check to see we weren't executed directly on the command line. It's a
	// crude method.
	if (argc != 1 || strncmp(argv[0], SINK_EXEC_NAME, strlen(SINK_EXEC_NAME))) {
		fprintf(stderr, "Don't execute localsink yourself.\n");
		return -1;
	}

	BUG(!environ);

	err = setup();
	if (err < 0)
		_exit(-1);

	// FIXME Instead of processing CUDA RPC here, we should just call sigsuspend
	// and spawn threads in the shm_callback for each new region, instead.

	// The shm_callback will modify this variable. If a region is registered, it
	// will enable it, and vice versa. We assume single-threaded cuda
	// applications, performing one single mkreg and rmreg.
	while (!process_rpc)
		;

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

	wait_for_termination();

	teardown(); // doesn't return
}

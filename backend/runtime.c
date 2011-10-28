/**
 * @file runtime.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-25
 *
 * @brief Entry point for applications to enter assembly runtime. Watch for
 * applications to register, accept shared memory locations and assign them
 * assemblies.
 *
 * FIXME This file is a mess. Clean it up later to keep track of all sinks it
 * spawns. And find a reasonable way to specify assembly hint structures/files.
 */

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <debug.h>
#include <common/libregistration.h>
#include <assembly.h>
#include "sinks.h"

// deal with only one assembly for now, then expand
// later we can store a list of PIDs we fork into local sinks with their
// respective assembly IDs
asmid_t asmid;
struct assembly_cap_hint *hint = NULL;

pid_t localsink_pid;

static int fork_localsink(asmid_t asmid, regid_t regid)
{
	pid_t pid;

	printd(DBG_INFO, "Forking localsink\n");

	pid = fork();
	if (pid == 0) { // we're the child process
		localsink(asmid, regid);
		printd(DBG_ERROR, "localsink returned!\n");
		assert(0); // localsink should NEVER return up out of localsink()
	} else if (pid != 0) { // we're the parent
		localsink_pid = pid;
		printd(DBG_DEBUG, "forked localsink, pid=%d\n",
				localsink_pid);
	} else {
		printd(DBG_ERROR, "fork() failed, pid=%d\n", pid);
		goto fail;
	}
	return 0;
fail:
	return -1;
}

static void registration_callback(enum callback_event e, regid_t regid)
{
	asmid_t asmid;
	void *region;
	size_t size;
	int err;

	struct assembly_cap_hint hint;
	memset(&hint, 0, sizeof(hint));
	hint.num_gpus = 1;

	switch (e) {
		// Process has indicated it wishes to join runtime.
		case CALLBACK_NEW:
			region = reg_be_get_shm(regid);
			size = reg_be_get_shm_size(regid);
			printd(DBG_DEBUG, "new reg; shm=%p sz=%lxB\n", region, size);
			asmid = assembly_request(&hint);
			if (!VALID_ASSEMBLY_ID(asmid)) {
				printd(DBG_ERROR, "Could not request assembly\n");
			}
			printd(DBG_INFO, "Acquired assembly: asmid=%lu size=%d\n",
					asmid, assembly_num_vgpus(asmid));
			err = fork_localsink(asmid, regid);
			if (err < 0)
				printd(DBG_ERROR, "Could not fork localsink\n");
			break;

		// Process has indicated it wishes to leave runtime.
		case CALLBACK_DEL:
			break;

		default:
			printd(DBG_ERROR, "Unknown callback event %u\n", e);
			break;
	}
}

int main(void)
{
	int err;

	// TODO Use command line to determine which node type we are.
	err = assembly_runtime_init(NODE_TYPE_MAIN);
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize assembly runtime\n");
		return -1;
	}

	// Initialize library <--> backend registration.
	err = reg_be_init(32); // # processes we expect to register
	if (err < 0) {
		fprintf(stderr, "Could not initialize library registration\n");
		return -1;
	}
	err = reg_be_callback(registration_callback);
	if (err < 0) {
		printd(DBG_ERROR, "Could not set registration callback\n");
		return -1;
	}

	// TODO Wait on all sink processes instead of sleeping.
	// TODO Install sig handler to capture premature exit.
	printf("Sleeping for some time\n");
	sleep(3600);

	// Close application registration.
	err = reg_be_shutdown();
	if (err < 0) {
		fprintf(stderr, "Could not shutdown library registration\n");
		return -1;
	}

	err = assembly_runtime_shutdown();
	if (err < 0) {
		printd(DBG_ERROR, "Could not shutdown assembly runtime\n");
		return -1;
	}

	return 0;
}

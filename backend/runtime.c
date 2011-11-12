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

// System includes
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Project includes
#include <shmgrp.h>
#include <debug.h>
#include <assembly.h>

// Directory-immediate includes
#include "sinks.h"

/*-------------------------------------- INTERNAL STATE ----------------------*/

// XXX We assume a single-process single-threaded CUDA application for now. Thus
// one assembly, one hint and one sink child.
static struct assembly_cap_hint hint;
static struct sink_child sink_child;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

static void runtime_entry(group_event e, pid_t pid)
{
	int err;
	asmid_t asmid;
	memset(&hint, 0, sizeof(hint));
	hint.num_gpus = 1;

	switch (e) {
		case MEMBERSHIP_JOIN:
		{
			// TODO in the future, detect multi-rank MPI applications, and
			// give them the same assembly. Perhaps provide this in the hint
			// structure.
			pid_t childpid;
			printd(DBG_INFO, "Process %d is requesting to join the assembly\n", pid);
			asmid = assembly_request(&hint);
			if (!VALID_ASSEMBLY_ID(asmid)) {
				printd(DBG_ERROR, "Error requesting assembly\n");
				break;
			}
			printd(DBG_INFO, "Acquired assembly %lu with %d GPUs\n",
					asmid, assembly_num_vgpus(asmid));
			err = fork_localsink(asmid, pid, &childpid);
			// XXX ONLY the parent should return from here!
			if (err < 0) {
				printd(DBG_ERROR, "Error forking localsink for assembly %lu\n",
						asmid);
				break;
			}
			sink_child.pid = childpid;
			sink_child.type = SINK_EXEC_LOCAL;
			sink_child.asmid = asmid;
		}
		break;
		case MEMBERSHIP_LEAVE:
		{
			int ret; // return val of child
			printd(DBG_INFO, "Application %d is leaving the runtime\n", pid);
			// Tell it to stop, then wait for it to disappear.
			err = kill(sink_child.pid, SINK_TERM_SIG);
			if (err < 0) {
				printd(DBG_ERROR, "Could not send signal %d to child %d\n",
						SINK_TERM_SIG, sink_child.pid);
			}
			err = waitpid(sink_child.pid, &ret, 0);
			if (err < 0) {
				printd(DBG_ERROR, "Could not wait on child %d\n",
						sink_child.pid);
			}
			printd(DBG_DEBUG, "Child exited with code %d\n", ret);
			// Now kill its assembly
			// XXX XXX XXX
			// Only kill its assembly IF there are no longer any processes in
			// its application group using the assembly! Not sure how to verify
			// this.
			// XXX XXX XXX
			err = assembly_teardown(sink_child.asmid);
			if (err < 0) {
				printd(DBG_ERROR, "Could not destroy assembly %lu\n",
						sink_child.asmid);
			}
		}
		break;
		default:
			printd(DBG_ERROR, "Error: invalid membership event %d\n", e);
			break;
	}
}

static void sigint_handler(int sig)
{
	; // Do nothing, just prevent it from killing us
}

static int start_runtime(void)
{
	int err;
	err = assembly_runtime_init(NODE_TYPE_MAIN); // TODO Take a command arg
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize assembly runtime\n");
		return -1;
	}
	err = shmgrp_init();
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize shmgrp state\n");
		return -1;
	}
	err = shmgrp_open(ASSEMBLY_SHMGRP_KEY, runtime_entry, true);
	if (err < 0) {
		printd(DBG_ERROR, "Could not open shmgrp %s\n", ASSEMBLY_SHMGRP_KEY);
		shmgrp_tini();
		return -1;
	}
	// FIXME Initialize a list of sinks instead of one child.
	memset(&sink_child, 0, sizeof(sink_child));
	return 0;
}

static void shutdown_runtime(void)
{
	int err;
	err = shmgrp_close(ASSEMBLY_SHMGRP_KEY);
	if (err < 0)
		printd(DBG_ERROR, "Could not close shmgrp\n");
	err = shmgrp_tini();
	if (err < 0)
		printd(DBG_ERROR, "Could not deallocate shmgrp state\n");
	err = assembly_runtime_shutdown();
	if (err < 0)
		printd(DBG_ERROR, "Could not shutdown assembly runtime\n");
	printd(DBG_INFO, "\nAssembly runtime shut down.\n");

}

/*-------------------------------------- ENTRY -------------------------------*/

int main(void)
{
	int err;
	sigset_t mask;
	struct sigaction action;

	// Block all signals.
	sigfillset(&mask);
	sigprocmask(SIG_BLOCK, &mask, NULL);

	err = start_runtime();
	if (err < 0)
		return -1;

	printd(DBG_INFO, "Assembly runtime ready to accept new CUDA applications.\n");

	// Install a new handler for SIGINT.
	memset(&action, 0, sizeof(action));
	action.sa_handler = sigint_handler;
	sigemptyset(&action.sa_mask);
	err = sigaction(SIGINT, &action, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Error installing signal handler for SIGINT.\n");
		shutdown_runtime();
		return -1;
	}

	// Atomically unblock SIGINT and wait for it.
	sigdelset(&mask, SIGINT);
	sigsuspend(&mask);

	shutdown_runtime();
	return 0;
}

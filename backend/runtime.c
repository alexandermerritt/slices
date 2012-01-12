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
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Project includes
#include <assembly.h>
#include <debug.h>
#include <shmgrp.h>
#include <util/timer.h>

// Directory-immediate includes
#include "sinks.h"

/*-------------------------------------- EXTERNAL VARIABLES ------------------*/

// Current process' environment variables (POSIX). Modified with putenv.
// man 7 environ
extern char **environ;

/*-------------------------------------- INTERNAL STATE ----------------------*/

// XXX We assume a single-process single-threaded CUDA application for now. Thus
// one assembly, one hint and one sink child.
static const struct assembly_hint hint =
{
	.num_gpus = 1,
	.nic_type = HINT_USE_ETH
};
static struct sink sink;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

int forksink(asmid_t asmid, pid_t memb_pid, pid_t *sink_pid)
{
	int err;

	*sink_pid = fork();

	// Fork error.
	if (*sink_pid < 0) {
		printd(DBG_ERROR, "fork failed: %s\n", strerror(errno));
		return -1;
	}

	// Parent does nothing more.
	if (*sink_pid > 0) {
		return 0;
	}

	// For some reason, the damn POSIX variable is set to NULL in the child
	// sometimes
	BUG(!environ);

	/*
	 * We're the child process. Add our environment variables needed to
	 * configure the sink, then exec the sink. It will read these and configure
	 * itself. Environment variables we add here don't affect the parent, since
	 * we already forked. Variables we set are preserved across exec*
	 *
	 * After this point, it is imperative that return is not used to quit.
	 * _exit() should be used instead, since we don't want to return up the
	 * forksink callstack.
	 */

	/* Time the work the new child performs before exec'ing into a localsink. */
	TIMER_DECLARE1(sink);
	TIMER_START(sink);

	// Need separate strings for the environment variables we add, because
	// putenv does NOT copy the strings, it merely sets pointers to these arrays
	char env_uuid[ENV_MAX_LEN];
	memset(env_uuid, 0, ENV_MAX_LEN);
	char env_pid[ENV_MAX_LEN];
	memset(env_pid, 0, ENV_MAX_LEN);

	// Export the assembly to get the UUID
	char uuid_str[64];
	assembly_key_uuid key;
	err = assembly_export(asmid, key);
	if (err < 0) {
		printd(DBG_ERROR, "Could not export assembly %lu\n", asmid);
		_exit(-1);
	}
	uuid_unparse(key, uuid_str);
	snprintf(env_uuid, ENV_MAX_LEN, "%s=%s", SINK_ASSM_EXPORT_KEY_ENV, uuid_str);
	err = putenv(env_uuid);
	if (err < 0) {
		printd(DBG_ERROR, "Could not add assembly key env var\n");
		_exit(-1);
	}

	// Set the application PID
	snprintf(env_pid, ENV_MAX_LEN, "%s=%d", SINK_SHMGRP_PID, memb_pid);
	err = putenv(env_pid);
	if (err < 0) {
		printd(DBG_ERROR, "Could not add member PID env var\n");
		_exit(-1);
	}

#ifdef TIMING
	uint64_t sink_setup;
   	TIMER_END(sink, sink_setup);
	printf(TIMERMSG_PREFIX "sink-setup %lu\n", sink_setup);
#endif

	// Go!
	err = execle("./localsink", SINK_EXEC_NAME, NULL, environ);
	if (err < 0) {
		printd(DBG_ERROR, "Could not execle: %s\n", strerror(errno));
		_exit(-1);
	}

	return -0xb00b; // satisfy the compiler gods
}

static void runtime_entry(group_event e, pid_t pid)
{
	int err;
	asmid_t asmid;

	TIMER_DECLARE1(timer);
#ifdef TIMING
	uint64_t timing;
#endif	/* TIMING */

	switch (e) {
		case MEMBERSHIP_JOIN:
		{
			// TODO detect process groups (e.g. MPI)
			pid_t childpid;
			printf("Process %d is joining the runtime.\n", pid);

			TIMER_START(timer);
			asmid = assembly_request(&hint);
			TIMER_END(timer, timing);
#ifdef TIMING
			printf(TIMERMSG_PREFIX "assm-request %lu\n", timing);
#endif

			if (!VALID_ASSEMBLY_ID(asmid)) {
				printd(DBG_ERROR, "Error requesting assembly\n");
				break;
			}
			assembly_print(asmid);

			TIMER_START(timer);
			err = forksink(asmid, pid, &childpid);
			if (err < 0) {
				printd(DBG_ERROR, "Could not fork\n");
				break;
			}
			TIMER_END(timer, timing);
#ifdef TIMING
			printf(TIMERMSG_PREFIX "fork %lu\n", timing);
#endif
			sink.pid = childpid;
			sink.type = SINK_EXEC_LOCAL;
			sink.asmid = asmid;
		}
		break;
		case MEMBERSHIP_LEAVE:
		{
			printf("Process %d is leaving the runtime.\n", pid);

			TIMER_START(timer);

			// Tell it to stop, then wait for it to disappear.
			int ret; // return val of child
			err = kill(sink.pid, SINK_TERM_SIG);
			if (err < 0) {
				printd(DBG_ERROR, "Could not send signal %d to child %d\n",
						SINK_TERM_SIG, sink.pid);
			}
			err = waitpid(sink.pid, &ret, 0);
			if (err < 0) {
				printd(DBG_ERROR, "Could not wait on child %d\n",
						sink.pid);
			}
			printd(DBG_DEBUG, "Child exited with code %d\n", ret);

			// Now kill its assembly
			// XXX XXX XXX
			// Only kill its assembly IF there are no longer any processes in
			// its application group using the assembly! Not sure how to verify
			// this.
			// XXX XXX XXX
			err = assembly_teardown(sink.asmid);
			if (err < 0) {
				printd(DBG_ERROR, "Could not destroy assembly %lu\n",
						sink.asmid);
			}

			TIMER_END(timer, timing);
#ifdef TIMING
			printf(TIMERMSG_PREFIX "leave %lu\n", timing);
#endif
		}
		break;
		default:
			printd(DBG_ERROR, "Error: invalid membership event %u\n", e);
			break;
	}
}

static void sigint_handler(int sig)
{
	; // Do nothing, just prevent it from killing us
}

static int start_runtime(enum node_type type, const char *main_ip)
{
	int err;
	err = shmgrp_init();
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize shmgrp state\n");
		return -1;
	}
	err = shmgrp_open(ASSEMBLY_SHMGRP_KEY, runtime_entry);
	if (err < 0) {
		printd(DBG_ERROR, "Could not open shmgrp %s\n", ASSEMBLY_SHMGRP_KEY);
		shmgrp_tini();
		return -1;
	}
	err = assembly_runtime_init(type, main_ip);
	if (err < 0) {
		printd(DBG_ERROR, "Could not initialize assembly runtime\n");
		return -1;
	}
	// FIXME Initialize a list of sinks instead of one child.
	memset(&sink, 0, sizeof(sink));
	return 0;
}

static void shutdown_runtime(void)
{
	int err;
	err = shmgrp_close(ASSEMBLY_SHMGRP_KEY);
	if (err < 0) {
		printd(DBG_ERROR, "Could not close shmgrp\n");
	}
	err = shmgrp_tini();
	if (err < 0) {
		printd(DBG_ERROR, "Could not deallocate shmgrp state\n");
	}
	err = assembly_runtime_shutdown();
	if (err < 0) {
		printd(DBG_ERROR, "Could not shutdown assembly runtime\n");
	}
	printf("\nAssembly runtime shut down.\n");

}

static bool verify_args(int argc, char *argv[], enum node_type *type)
{
	const char main_str[] = "main";
	const char minion_str[] = "minion";
	if (!argv)
		return false;
	if (argc < 2 || argc > 3)
		return false;
	if (argc == 2) { // ./runtime main
		if (strncmp(argv[1], main_str, strlen(main_str)) != 0)
			return false;
		*type = NODE_TYPE_MAIN;
	} else if (argc == 3) { // ./runtime minion <ip-addr>
		if (strncmp(argv[1], minion_str, strlen(minion_str)) != 0)
			return false;
		// TODO verify ip via regex
		*type = NODE_TYPE_MINION;
	}
	return true;
}

static void print_usage(void)
{
	const char usage_str[] =
		"Usage: ./runtime main\n"
		"       ./runtime minion <ip-addr>\n";
	fprintf(stderr, usage_str);
}

/*-------------------------------------- ENTRY -------------------------------*/

int main(int argc, char *argv[])
{
	int err = 0;
	sigset_t mask;
	struct sigaction action;
	enum node_type type;
	const char start_msg[] =
		"Assembly runtime up. Start other participants and/or CUDA applications.\n"
		"Type [Ctrl+c] or issue [kill -s SIGINT] to pid %d to shutdown.\n";

#ifdef DEBUG
	printf("(Built with debug symbols)\n");
#endif

	if (!verify_args(argc, argv, &type)) {
		print_usage();
		return -1;
	}

	if (argc == 2)
		err = start_runtime(type, NULL);
	else if (argc == 3)
		err = start_runtime(type, argv[2]);
	if (err < 0) {
		fprintf(stderr, "Could not initialize. Check your arguments.\n");
		return -1;
	}

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

	printf(start_msg, getpid());

	// Wait for any signal.
	sigemptyset(&mask);
	sigsuspend(&mask);

	shutdown_runtime();
	return 0;
}

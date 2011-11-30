/**
 * @file localsink.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-27
 *
 * @brief TODO
 */

// System includes
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <string.h>

// Other project includes
#include <shmgrp.h>

// Project includes
#include <assembly.h>
#include <debug.h>
#include <util/compiler.h>
#include <util/list.h>
#include <util/x86_system.h>

// Directory-immediate includes
#include "sinks.h"

/*-------------------------------------- EXTERNAL VARIABLES ------------------*/

//! POSIX environment variable list, man 7 environ
extern char **environ;

/*-------------------------------------- INTERNAL DEFINITIONS ----------------*/

/**
 * Thread state associated with a 'proxy' thread. Proxies map 1:1 with
 * application threads to execute RPCs in their own shm regions.
 */
struct proxy_state
{
	struct list_head link; //! Link together all proxy threads into a list
	struct shmgrp_region region; //! Memory region mapped with the app thread
	bool exit_loop; //! Used by proxy_halt to cause a thread to break its loop

	bool is_alive;
	int exit_code;
	pthread_t tid; //! ID of proxy thread itself
};

/**
 * Global state associated with this instance of localsink.
 */
struct internals_state
{
	pthread_mutex_t lock; //! Lock for changes to internal state
	pid_t pid; //! The application PID we're working for
	assembly_key_uuid uuid; //! Assembly UUID key used for importing
	asmid_t asmid; //! Assembly the app was given, which we've imported
	struct list_head proxy_list; //! List of all proxy threads
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct internals_state *internals = NULL;

/*-------------------------------------- INTERNAL THREADING ------------------*/

static inline void
proxy_add(struct list_head *list, struct proxy_state *proxy)
{
	list_add(&proxy->link, list);
}

static inline void
proxy_rm(struct proxy_state *proxy)
{
	list_del(&proxy->link);
}

/**
 * Proxy thread "destructor".
 */
static void
proxy_cleanup(void *arg)
{
	struct proxy_state *state = (struct proxy_state*)arg;
	state->is_alive = false;
	printd(DBG_DEBUG, "exited with code %d\n", state->exit_code);
	// TODO Anything else?
}

/**
 * Proxy thread entry point.
 */
static void *
proxy_thread(void *arg)
{
	int err, old_thread_state /* not used */;
	struct proxy_state *state = (struct proxy_state*)arg;
	struct cuda_packet *shmpkt = state->region.addr;
	int vgpu_id = 0; //! vgpu this thread maps to

	pthread_cleanup_push(proxy_cleanup, state);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	state->is_alive = true;

	while (!state->exit_loop) {

		// Check packet flag. If not ready, check loop flag. Spin.
		rmb();
		if (!(shmpkt->flags & CUDA_PKT_REQUEST)) {
			rmb(); continue;
		}

		// Update our vgpu association if the application thread does so.
		//if (unlikely(shmpkt->method_id == CUDA_SET_DEVICE))
		if (shmpkt->method_id == CUDA_SET_DEVICE)
			vgpu_id = shmpkt->args[0].argll;

		// Execute the RPC.
		err = assembly_rpc(internals->asmid, vgpu_id, shmpkt);
		if (err < 0) {
			printd(DBG_ERROR, "assembly_rpc returned %d\n", err);
			// what else to do? this is supposed to be an error of the assembly
			// module, not an error of the call specifically
			assert(0);
		}

		// Don't forget! Indicate packet is executed and on return path.
		shmpkt->flags = CUDA_PKT_RESPONSE;
		wmb();
	}

	pthread_cleanup_pop(1); // invoke cleanup routine
	pthread_exit(NULL);
}

static int
proxy_spawn(pid_t app_pid, shmgrp_region_id id)
{
	int err;
	struct proxy_state *proxy;

	// we'll assume the caller is responsible for ensuring threads do not share
	// the same memory regions

	proxy = malloc(sizeof(*proxy));
	if (!proxy)
		return -ENOMEM;
	INIT_LIST_HEAD(&proxy->link);
	proxy->is_alive = false;
	proxy->exit_code = 0;
	proxy->exit_loop = false;
	err = shmgrp_member_region(ASSEMBLY_SHMGRP_KEY, app_pid, id, &proxy->region);
	if (err < 0) return -1;
	err = pthread_create(&proxy->tid, NULL, proxy_thread, proxy);
	if (err < 0) return -1;
	// we'll join the thread later

	pthread_mutex_lock(&internals->lock);
	proxy_add(&internals->proxy_list, proxy);
	pthread_mutex_unlock(&internals->lock);

	return 0;
}

static void
proxy_halt(shmgrp_region_id id)
{
	int err;
	struct proxy_state *proxy = NULL;
	pthread_mutex_lock(&internals->lock);
	// find the proxy responsible for the given region
	list_for_each_entry(proxy, &internals->proxy_list, link)
		if (proxy->region.id == id)
			break;
	if (!proxy || proxy->region.id != id) {
		printd(DBG_WARNING, "proxy not found for region %d\n", id);
		pthread_mutex_unlock(&internals->lock);
		return;
	}
	// terminator it
	proxy->exit_loop = true;
	err = pthread_cancel(proxy->tid);
	if (err < 0) {
		printd(DBG_WARNING, "pthread_cancel: %s\n", strerror(errno));
	}
	err = pthread_join(proxy->tid, NULL);
	if (err < 0) {
		printd(DBG_WARNING, "pthread_join: %s\n", strerror(errno));
	}
	proxy_rm(proxy); // remove it from the list
	free(proxy);
	pthread_mutex_unlock(&internals->lock);
}

/*-------------------------------------- INTERNAL MGMT FUNCTIONS -------------*/

// The interposer allocates a region for each thread it detects within the
// application. Thus we assume the creation of a memory region also means a
// thread has been created in the application.
static void
region_request(shm_event e, pid_t memb_pid, shmgrp_region_id id)
{
	int err;
	printd(DBG_DEBUG, "PID %d %sing region %d\n", memb_pid,
			(e == SHM_CREATE_REGION ? "creat" : "destroy"), id);
	switch (e) {
		case SHM_CREATE_REGION:
			err = proxy_spawn(memb_pid, id);
			if (err < 0) {
				printd(DBG_ERROR, "failed to spawn proxy thread\n");
			}
			break;
		case SHM_REMOVE_REGION:
			proxy_halt(id);
			break;
		default:
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
	assembly_key_uuid uuid;
	pid_t pid;
	asmid_t asmid;
	
	//
	// Extract environment variables we're configured with.
	//

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

	//
	// Configure assembly state.
	//

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

	//
	// Initialize threading state (internals).
	//

	internals = calloc(1, sizeof(*internals));
	if (!internals) {
		fprintf(stderr, "Out of memory\n");
		return -ENOMEM;
	}
	pthread_mutex_init(&internals->lock, NULL);
	internals->pid = pid;
	memcpy(internals->uuid, uuid, sizeof(uuid));
	internals->asmid = asmid;
	INIT_LIST_HEAD(&internals->proxy_list);

	//
	// Configure shmgrp state.
	//

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

	//
	// Initialize remaining state.
	//

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
	// FIXME Halt all proxy threads
	err = shmgrp_destroy_member(ASSEMBLY_SHMGRP_KEY, internals->pid);
	if (err < 0) {
		printd(DBG_WARNING, "could not remove member PID %d\n", internals->pid);
	}
	err = __shmgrp_dump_group(ASSEMBLY_SHMGRP_KEY);
	if (err < 0) {
		printd(DBG_WARNING, "__shmgrp_dump_group failed\n");
	}
	err = assembly_runtime_shutdown();
	if (err < 0) {
		printd(DBG_WARNING, "__shmgrp_dump_group failed\n");
	}
	err = shmgrp_tini();
	if (err < 0) {
		printd(DBG_WARNING, "__shmgrp_dump_group failed\n");
	}
	if (!list_empty(&internals->proxy_list)) {
		printd(DBG_WARNING, "proxy threads still exist\n");
	}
	if (internals)
		free(internals);
	internals = NULL;
	_exit(0);
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

/**
 * Entry point for the local sink process. It is forked/exec'd from the runtime
 * when a new application process registers.
 *
 * This file shall NOT call any functions that request/destroy assemblies, nor
 * those that modify shmgrp state EXCEPT to establish/destroy members. From the
 * shmgrp perspective, it is a leader, of the same group the runtime is, but
 * will receive no notifications of joining members (via that double-underscore
 * function call, instead of actually opening a new group altogether).
 */
int main(int argc, char *argv[])
{
	int err;

	// Check to see we weren't executed directly on the command line.
	if (argc != 1 || strncmp(argv[0], SINK_EXEC_NAME, strlen(SINK_EXEC_NAME))) {
		fprintf(stderr, "Don't execute localsink yourself.\n");
		return -1;
	}

	err = setup();
	if (err < 0)
		_exit(-1);

	wait_for_termination();

	teardown(); // doesn't return
}

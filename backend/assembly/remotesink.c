/**
 * @file backend/assembly/remote.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief This file enables receiving work on a remote data path.
 *
 * FIXME MUCH of the code in this file and in assembly/rpc.c is duplicated, as
 * both implement RPC servers. Perhaps there's a clean way to share code?
 */

// System includes
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// Project includes
#include <assembly.h>
#include <cuda/ops.h>
#include <debug.h>
#include <fatcubininfo.h>
#include <io/sock.h>
#include <packetheader.h>

// Directory-immediate includes
#include "remote.h"
#include "types.h"

/*-------------------------------------- INTERNAL DEFINITIONS ----------------*/

/**
 * TODO
 */
struct payload
{
	void *buffer;
	size_t size;
};

/**
 * State associated with each thread. One thread is the admission thread, the
 * remainder are minions that service CUDA RPC requests from remote nodes,
 * implementing a remote vgpu within an assembly.
 */
struct rpc_thread
{
	// state unique to only minion threads
	struct list_head link;
	char hostname[HOST_LEN]; //! remote host connected
	//asmid_t asmid; // assembly ID this data path implements
	struct fatcubins *cubin; //! points to state in remote_cubin
	struct payload payload;	//! like the shm region in the interposer

	// state common to the admin and minion threads
	bool is_alive;
	pthread_t tid;
	int exit_code;
	struct sockconn conn;
};

/**
 * Each application process that exists in a system which implements an assembly
 * containing remote vgpus must cache duplicate copies of CUBIN data (created
 * via the __cudaRegister* function calls) at each node hosting a vgpu. We need
 * to distinguish among incoming connections so as to ensure that each instance
 * of CUBIN state is maintained once per process in the cluster (we assume a
 * process may have its own assembly, or is part of an MPI application with
 * multiple processes throughout a cluster, each using the same assembly).
 *
 * We cannot simply give each minion thread its own fatcubins* state because a
 * single-process application on one node may map all its vgpus in an assembly
 * to the same remote host. This situation necessitates only one CUBIN state
 * instance. However, each minion thread will use the assembly ID and hostname
 * it represents to look up if a CUBIN instance exists.
 *
 * If remotesink were to spawn processes for each incoming connection instead,
 * then we wouldn't have to have such complicated demuxing. Maybe if CUDA didn't
 * suck so much...
 *
 * TODO For now we will NOT handle two+ processes on a single node using the
 * same assembly which map the same remote vgpu instance.
 */
struct remote_cubin
{
	struct list_head link;
	struct fatcubins *cubins;
	char hostname[HOST_LEN];
	asmid_t asmid;
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct rpc_thread *admin_thread = NULL;

//! List of rpc_thread representing remote nodes connecting to us, as they host
//! an assembly with vgpus mapped to GPUs in our node. A "minion" is a thread
//! that is assigned to a remote host.
static struct list_head minions = LIST_HEAD_INIT(minions);
static pthread_mutex_t minion_lock = PTHREAD_MUTEX_INITIALIZER;

static struct list_head cubins = LIST_HEAD_INIT(cubins);
//static pthread_mutex_t cubin_lock = PTHREAD_MUTEX_INITIALIZER;

// FIXME Remove this when the rest of the code works, and we can support/lookup
// multiple CUBINs.
static struct fatcubins singleton_cubin;

/*-------------------------------------- INTERNAL THREADING ------------------*/

static inline void
__add_minion(struct list_head *minions, struct rpc_thread *state)
{
	list_add(&state->link, minions);
}

static inline void
__rm_minion(struct rpc_thread *state)
{
	list_del(&state->link);
}

#if 0
static inline void
__add_cubin(struct list_head *cubins, struct remote_cubin *cubin)
{
	list_add(&cubin->link, cubins);
}

static inline void
__rm_cubin(struct remote_cubin *cubin)
{
	list_del(&cubin->link);
}
#endif

static inline void
halt_rpc_thread(struct rpc_thread *state)
{
	int err;
	if (state->is_alive) {
		err = pthread_cancel(state->tid);
		if (err < 0) {
			printd(DBG_WARNING, "pthread_cancel: %s\n", strerror(errno));
		}
		err = pthread_join(state->tid, NULL);
		if (err < 0) {
			printd(DBG_WARNING, "pthread_join: %s\n", strerror(errno));
		}
	}
}

static void
minion_cleanup(void *arg)
{
	int err;
	struct rpc_thread *state = (struct rpc_thread*)arg;
	
	state->is_alive = false;

#define EXIT_STRING "cudarpc minion exit: "
	switch (state->exit_code) {
		case 0:
			printd(DBG_INFO, EXIT_STRING "no error\n");
			break;
		case -ENETDOWN:
			printd(DBG_ERROR, EXIT_STRING "error: network down\n");
			break;
		case -ENOMEM:
			printd(DBG_ERROR, EXIT_STRING "error: no memory\n");
			break;
		case -EPROTO:
			printd(DBG_ERROR, EXIT_STRING "error: thread spawn\n");
			break;
		default:
			BUG(1);
			break;
	}
	err = conn_close(&state->conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: close sock conn\n");
	}
#undef EXIT_STRING

	pthread_mutex_lock(&minion_lock);
	__rm_minion(state);
	pthread_mutex_unlock(&minion_lock);

	// FIXME If this thread is the last to use a certain cubin, remove the cubin

	printd(DBG_DEBUG, "Freeing buffers\n");
	if (state->payload.buffer)
		free(state->payload.buffer);
	printd(DBG_DEBUG, "Freeing state\n");
	free(state);
}

// forward declaration
static int do_cuda_rpc(struct sockconn*, struct payload*, struct fatcubins*);

static void *
minion_thread(void *arg)
{
	int err, old_thread_state /* not used */;
	struct rpc_thread *state = (struct rpc_thread*) arg;
	struct sockconn *conn = &state->conn;
	//struct fatcubins *_cubins = NULL;

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	pthread_cleanup_push(minion_cleanup, state);
	state->is_alive = true;

	// TODO Get the hostname from the socket.

	printd(DBG_INFO, "vgpu mapped in from TODO\n");

	state->payload.size = (128 << 20); // FIXME Don't hard code this
	state->payload.buffer = malloc(state->payload.size);
	if (!state->payload.buffer) {
		state->exit_code = -ENOMEM;
		pthread_exit(NULL);
	}

	while (1) {
		err = do_cuda_rpc(conn, &state->payload, &singleton_cubin);
		if (err < 0) {
			state->exit_code = err;
			break;
		}
	}
	pthread_cleanup_pop(1); // invoke cleanup routine
	pthread_exit(NULL);
}

static void
admission_cleanup(void *arg)
{
	int err;
	struct rpc_thread *state = (struct rpc_thread*)arg;
	
	state->is_alive = false;

#define EXIT_STRING "cudarpc admin exit: "
	switch (state->exit_code) {
		case 0:
			printd(DBG_INFO, EXIT_STRING "no error\n");
			break;
		case -ENETDOWN:
			printd(DBG_ERROR, EXIT_STRING "error: network down\n");
			break;
		case -ENOMEM:
			printd(DBG_ERROR, EXIT_STRING "error: no memory\n");
			break;
		case -EPROTO:
			printd(DBG_ERROR, EXIT_STRING "error: thread spawn\n");
			break;
		default:
			BUG(1);
			break;
	}
	err = conn_close(&state->conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: close sock conn\n");
	}
#undef EXIT_STRING

	// FIXME Check to see if we have other minion threads. Keep in mind

	printd(DBG_DEBUG, "Freeing state\n");
	free(state);
}

static void *
admission_thread(void *arg)
{
	int err, old_thread_state /* not used */;
	struct rpc_thread *state = (struct rpc_thread*) arg;
	struct sockconn *conn = &state->conn;

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	pthread_cleanup_push(admission_cleanup, state);
	state->is_alive = true;

	cubins_init(&singleton_cubin);

	err = conn_localbind(conn, REMOTE_CUDA_PORT);
	if (err < 0) {
		state->exit_code = -ENETDOWN;
		pthread_exit(NULL);
	}

	while (1) {
		struct sockconn new_conn;
		struct rpc_thread *new_state;
		err = conn_accept(conn, &new_conn);
		if (err < 0) {
			BUG(err == -EINVAL);
			state->exit_code = -ENETDOWN;
			break;
		}
		new_state = calloc(1, sizeof(*new_state));
		if (!new_state) {
			state->exit_code = -ENOMEM;
			break;
		}
		new_state->conn = new_conn;
		INIT_LIST_HEAD(&new_state->link);
		//new_state->hostname = ????? FIXME
		new_state->cubin = &singleton_cubin;
		// minion will initialize payload itself
		err = pthread_create(&new_state->tid, NULL, minion_thread, new_state);
		if (err < 0) {
			state->exit_code = -EPROTO;
		}
		err = pthread_detach(new_state->tid); // we'll never pthread_join it
		if (err < 0) {
			state->exit_code = -EPROTO;
			break;
		}
		pthread_mutex_lock(&minion_lock);
		__add_minion(&minions, new_state);
		pthread_mutex_unlock(&minion_lock);
	}
	pthread_cleanup_pop(1); // invoke cleanup routine
	pthread_exit(NULL);
}

/*-------------------------------------- INTERNAL RPC FUNCTIONS --------------*/

// Copied from assembly.c FIXME Is there some way this code can be shared?
static int
demux(struct cuda_packet *pkt, struct fatcubins *cubins)
{
	switch (pkt->method_id) {

		// Functions which take only a cuda_packet*
		case CUDA_CONFIGURE_CALL:
			exec_ops.configureCall(pkt);
			break;
		case CUDA_FREE:
			exec_ops.free(pkt);
			break;
		case CUDA_MALLOC:
			exec_ops.malloc(pkt);
			break;
		case CUDA_MEMCPY_D2D:
			exec_ops.memcpyD2D(pkt);
			break;
		case CUDA_MEMCPY_D2H:
			exec_ops.memcpyD2H(pkt);
			break;
		case CUDA_MEMCPY_H2D:
			exec_ops.memcpyH2D(pkt);
			break;
		case CUDA_SET_DEVICE:
			exec_ops.setDevice(pkt);
			break;
		case CUDA_SETUP_ARGUMENT:
			exec_ops.setupArgument(pkt);
			break;
		case CUDA_THREAD_EXIT:
			exec_ops.threadExit(pkt);
			break;
		case CUDA_THREAD_SYNCHRONIZE:
			exec_ops.threadSynchronize(pkt);
			break;
		case __CUDA_UNREGISTER_FAT_BINARY:
			exec_ops.unregisterFatBinary(pkt);
			break;

			// Functions which take a cuda_packet* and fatcubins*
		case CUDA_LAUNCH:
			exec_ops.launch(pkt, cubins);
			break;
		case CUDA_MEMCPY_FROM_SYMBOL_D2H:
			exec_ops.memcpyFromSymbolD2H(pkt, cubins);
			break;
		case CUDA_MEMCPY_TO_SYMBOL_H2D:
			exec_ops.memcpyToSymbolH2D(pkt, cubins);
			break;
		case __CUDA_REGISTER_FAT_BINARY:
			exec_ops.registerFatBinary(pkt, cubins);
			break;
		case __CUDA_REGISTER_FUNCTION:
			exec_ops.registerFunction(pkt, cubins);
			break;
		case __CUDA_REGISTER_VARIABLE:
			exec_ops.registerVar(pkt, cubins);
			break;

		default:
			printd(DBG_ERROR, "Method %d not supported in demux\n",
					pkt->method_id);
			goto fail;
	}
	return 0;
fail:
	return -1;
}

typedef enum {TO_HOST = 1, FROM_HOST} data_direction;

/**
 * Determine if an RPC has payload data associated with it (pointer arguments).
 * If so, return the direction the payload flows (input or output pointer
 * argument), and how large it is.  This function aims to assist do_cuda_rpc in
 * being able to be generic enough to handle any CUDA RPC.
 *
 * TODO This function could be placed into some external file, if needed. It
 * doesn't depend on any file-resident state.
 */
static bool
cudarpc_has_payload(
		const struct cuda_packet *pkt,	//! the RPC to inspect
		data_direction *direction,		//! which way it's flowing
		size_t *size)					//! how much data
{
	bool has_payload = true;

	/* XXX XXX XXX The code in this function GREATLY depends on how the
	 * functions in the interposer are coded, as we look at specific arguments
	 * for meaningful data. It'd be cleaner to have some nice method for
	 * specifying payload parameters and size in the packet generically, instead
	 * of this damn jump table.
	 */
	switch (pkt->method_id) {
		case CUDA_GET_DEVICE_PROPERTIES:
			*size = sizeof(struct cudaDeviceProp);
			*direction = TO_HOST;
			break;
		case CUDA_SETUP_ARGUMENT:
			*size = pkt->args[1].arr_argi[0];
			*direction = FROM_HOST;
			break;
		case CUDA_MEMCPY_H2D:
			*size = pkt->args[2].arr_argi[0];
			*direction = FROM_HOST;
			break;
		case CUDA_MEMCPY_D2H:
			*size = pkt->args[2].arr_argi[0];
			*direction = TO_HOST;
			break;
		case CUDA_MEMCPY_TO_SYMBOL_H2D:
			*size = pkt->args[2].arr_argi[0];
			*direction = FROM_HOST;
			break;
		case CUDA_MEMCPY_FROM_SYMBOL_D2H:
			*size = pkt->args[2].arr_argi[0];
			*direction = TO_HOST;
			break;
		case  __CUDA_REGISTER_FAT_BINARY:
			*size = pkt->args[1].argll;
			*direction = FROM_HOST;
			break;
		case __CUDA_REGISTER_FUNCTION:
			*size = pkt->args[1].arr_argi[0];
			*direction = FROM_HOST;
			break;
		case __CUDA_REGISTER_VARIABLE:
			*size = pkt->args[1].arr_argi[0];
			*direction = FROM_HOST;
			break;
		default: // everything else has no data, or is not a supported call
			has_payload = false;
			break;
	}

	return has_payload;
}

/**
 * Receive, process and dismiss one CUDA RPC. The goal is to set up the memory
 * region in a way that common/cuda/execute.c expects it, which is how the
 * interposer sets it up. This function is thread-safe as it operates only on
 * data accessible via the parameters.
 *
 * @param paylod	memory region used for receiving and sending data, exactly
 *					like the shm region used by the interposer
 */
static int
do_cuda_rpc(
		struct sockconn *conn,		//! network connection to use
		struct payload *payload,	//! contiguous mem reg storing pkt + data
		struct fatcubins *_cubins)	//! CUBIN state necessary for symbol lookup
{
	int err;
	struct cuda_packet *pkt = payload->buffer; //! pkt placed at top of region

	BUG(!pkt);

	bool has_payload; //! any data an RPC requires is stored after the pkt
	data_direction direction;
	size_t data_size;

	// Transactions always start with the packet itself. This allows both ends
	// to determine the remainder of the protocol from the method_id directly.
	err = conn_get(conn, pkt, sizeof(*pkt));
	if (err < 0) return -ENETDOWN;

	has_payload = cudarpc_has_payload(pkt, &direction, &data_size);

	if (has_payload && direction == FROM_HOST) {
		// TODO realloc buffer if size greater
		err = conn_get(conn, (pkt + 1), data_size);
		if (err < 0) return -ENETDOWN;
	}

	err = demux(pkt, _cubins);
	if (err < 0) return -1;

	// Always return the packet. Some RPCs don't need anything else.
	err = conn_put(conn, pkt, sizeof(*pkt));
	if (err < 0) return -ENETDOWN;

	if (has_payload && direction == TO_HOST) {
		err = conn_put(conn, (pkt + 1), data_size);
		if (err < 0) return -ENETDOWN;
	}
	return 0;
}

/*-------------------------------------- INTERNAL PROCESS FUNCTIONS ----------*/

static void
sigterm_handler(int sig)
{
	printf("remotesink caught sigterm\n");
	fflush(stdout);
	; // Do nothing, just prevent it from killing us
}

static int
setup(void)
{
	int err;

	admin_thread = calloc(1, sizeof(*admin_thread));
	if (!admin_thread)
		return -1;

	err = pthread_create(&admin_thread->tid, NULL, admission_thread, admin_thread);
	if (err < 0)
		return -1;

	// Install a handler for our term sig so it doesn't kill us
	struct sigaction action;
	memset(&action, 0, sizeof(action));
	action.sa_handler = sigterm_handler;
	sigemptyset(&action.sa_mask);
	err = sigaction(REMOTE_TERM_SIG, &action, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Could not install sig handler.\n");
		return -1;
	}

	return 0;
}

static void
wait_for_termination(void)
{
	int err;
	sigset_t mask;
	sigfillset(&mask);
	sigdelset(&mask, REMOTE_TERM_SIG);
	err = sigsuspend(&mask);
	if (err < 0)
		printd(DBG_ERROR, "sigsuspend returned error\n");
}

// Tell the compiler all calls to teardown never return.
static void teardown(void) __attribute__((noreturn));

static void
teardown(void)
{
	halt_rpc_thread(admin_thread);
	_exit(0);
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int main(int argc, char *argv[])
{
	int err;

	// Check to see we weren't executed directly on the command line. It's a
	// crude method.
	int name_size = strlen(REMOTE_EXEC_NAME);
	if (strncmp(argv[0], REMOTE_EXEC_NAME, name_size)) {
		fprintf(stderr, "Don't execute remotesink yourself.\n");
		exit(-1);
	}

	err = setup();
	if (err < 0)
		_exit(-1);

	printd(DBG_INFO, "waiting for termination\n");
	wait_for_termination();
	printd(DBG_INFO, "tearing down\n");

	teardown(); // doesn't return
}

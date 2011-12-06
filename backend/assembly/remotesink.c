/**
 * @file backend/assembly/remote.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief This file enables receiving work on a remote data path. It is
 * technically an "extension" to the assembly module, but exists within another
 * file. It does not have access to, nor need access to the assembly API, but
 * operates as its remote limb.
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
#include <util/compiler.h>

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
 * instance. However, each minion thread will use the PID and hostname it
 * represents to look up if a CUBIN instance exists (a minion more likely
 * represents a thread in an application, but a set of minions coming from the
 * same process somewhere share the same CUBIN state).
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
	struct list_head link; //! Link all remote_cubins together
	struct fatcubins cubins; //! CUBIN state
	char hostname[HOST_LEN]; //! Node hosting the CUDA application
	pid_t pid; //! CUDA application PID
	int ref_count; //! Number of minion threads using this state

	// XXX HACK I should instead properly maintain fatcubins.num_cubins
	// This counter is incremented with new registerFB calls, and decremented
	// with unregisterFB calls. Once it reaches zero, a minion thread stops.
	int reg_count;
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
	pid_t pid; //! localsink process attached to us; used to lookup cubin state
	struct remote_cubin *rcubin; //! (shared) cubin state with other minions
	struct payload payload;	//! packet and data; like the interposer shm region

	// state common to the admin and minion threads
	bool is_alive;
	pthread_t tid;
	int exit_code;
	struct sockconn conn;
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct rpc_thread *admin_thread = NULL;

//! List of rpc_thread representing remote nodes connecting to us, as they host
//! an assembly with vgpus mapped to GPUs in our node. A "minion" is a thread
//! that is assigned to a remote host.
static struct list_head minions = LIST_HEAD_INIT(minions);
static pthread_mutex_t minion_lock = PTHREAD_MUTEX_INITIALIZER;

static struct list_head rcubins = LIST_HEAD_INIT(rcubins); //! remote_cubin list
static pthread_mutex_t rcubin_lock = PTHREAD_MUTEX_INITIALIZER;

/*-------------------------------------- CUDA CUBIN FUNCTIONS ----------------*/

// __ functions are only to be called by non-__ functions
// Code here should call the latter directly, not the former.

static inline bool
__rcubin_match(const struct remote_cubin *rcubin, pid_t pid, const char *hostname)
{
	bool same_pid = (rcubin->pid == pid);
	bool same_host = (strncmp(rcubin->hostname, hostname, HOST_LEN) == 0);
	return (same_pid && same_host);
}

static inline void
__rcubin_add(struct list_head *rcubins, struct remote_cubin *rcubin)
{
	list_add(&rcubin->link, rcubins);
}

static inline void
__rcubin_rm(struct remote_cubin *rcubin)
{
	list_del(&rcubin->link);
}

static struct remote_cubin *
__rcubin_alloc_insert(struct list_head *rcubins, pid_t pid, const char *hostname)
{
	struct remote_cubin *rcubin;
	rcubin = malloc(sizeof(*rcubin));
	if (!rcubin)
		return NULL;
	cubins_init(&rcubin->cubins);
	strncpy(rcubin->hostname, hostname, HOST_LEN);
	rcubin->pid = pid;
	rcubin->ref_count = 1;
	rcubin->reg_count = 0;
	INIT_LIST_HEAD(&rcubin->link);
	__rcubin_add(rcubins, rcubin);
	return rcubin;
}

//! New minion thread spawned (new app thread connected): lookup CUBIN state; if
//! not found, create/insert new one & return that.
static struct remote_cubin *
rcubin_lookup(struct list_head *rcubins, pid_t pid, const char *hostname)
{
	bool found = false;
	struct remote_cubin *rcubin = NULL;
	list_for_each_entry(rcubin, rcubins, link) {
		if (__rcubin_match(rcubin, pid, hostname)) {
			found = true;
			break;
		}
	}
	if (!found)
		return __rcubin_alloc_insert(rcubins, pid, hostname);
	rcubin->ref_count++;
	return rcubin;
}

//! Minion thread is leaving (app thread exiting): if reference count is zero,
//! delete the rcubin state.
static int
rcubin_depart(struct list_head *rcubins, pid_t pid, const char *hostname)
{
	bool found = false;
	struct remote_cubin *rcubin = NULL;
	list_for_each_entry(rcubin, rcubins, link) {
		if (__rcubin_match(rcubin, pid, hostname)) {
			found = true;
			break;
		}
	}
	if (!found)
		return -EINVAL;
	rcubin->ref_count--;
	if (rcubin->ref_count == 0) {
		// FIXME Call cubins_dealloc
		__rcubin_rm(rcubin);
		free(rcubin);
	}
	return 0;
}

/*-------------------------------------- INTERNAL THREADING ------------------*/

static inline void
minion_add(struct list_head *minions, struct rpc_thread *state)
{
	list_add(&state->link, minions);
}

static inline void
minion_rm(struct rpc_thread *state)
{
	list_del(&state->link);
}

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
	minion_rm(state);
	pthread_mutex_unlock(&minion_lock);

	// Indicate we are no longer using CUBIN state
	pthread_mutex_lock(&rcubin_lock);
	err = rcubin_depart(&rcubins, state->pid, state->hostname);
	if (err < 0) {
		printd(DBG_WARNING, "rcubin_depart returned %d\n", err);
	}
	pthread_mutex_unlock(&rcubin_lock);

	if (state->payload.buffer)
		free(state->payload.buffer);
	free(state);
	printd(DBG_INFO, "exiting\n");
}

// forward declaration
static int do_cuda_rpc(struct sockconn*, struct payload*, struct remote_cubin*);

static void *
minion_thread(void *arg)
{
	int err, old_thread_state /* not used */;
	struct rpc_thread *state = (struct rpc_thread*) arg;
	struct sockconn *conn = &state->conn;

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	pthread_cleanup_push(minion_cleanup, state);
	state->is_alive = true;

	// Extract the remote peer's hostname
	err = conn_peername(&state->conn, state->hostname);
	if (err < 0) {
		state->exit_code = -ENETDOWN;
		pthread_exit(NULL);
	}

	// Pull out the PID of the localsink sending us RPCs. When an assembly is
	// mapped, this first piece of information is sent across. We can use the
	// localsink PID instead of the application PID so long as they exist in a
	// 1:1 ratio.
	err = conn_get(&state->conn, &state->pid, sizeof(state->pid));
	if (err < 0) {
		state->exit_code = -ENETDOWN;
		pthread_exit(NULL);
	}

	BUG(state->pid <= 0);

	// Get CUBIN state associated with new minion
	pthread_mutex_lock(&rcubin_lock);
	state->rcubin = rcubin_lookup(&rcubins, state->pid, state->hostname);
	pthread_mutex_unlock(&rcubin_lock);
	if (!state->rcubin) {
		state->exit_code = -ENOMEM;
		pthread_exit(NULL);
	}

	printd(DBG_INFO, "outbound vgpu mapped to us from PID %d on %s\n",
			state->pid, state->hostname);

	state->payload.size = (64 << 20); // FIXME Don't hard code this
	state->payload.buffer = malloc(state->payload.size);
	if (!state->payload.buffer) {
		state->exit_code = -ENOMEM;
		pthread_exit(NULL);
	}

	while (1) {
		err = do_cuda_rpc(conn, &state->payload, state->rcubin);
		if (unlikely(err < 0)) {
			if (unlikely(err == -ECANCELED)) {
				break; // client app finished w/o error, we stop processing
			}
			// everything else is a real error
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

	// FIXME Check to see if we have other minion threads.

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

		new_state->conn = new_conn; // whole struct copy
		INIT_LIST_HEAD(&new_state->link);

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
		minion_add(&minions, new_state);
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

/**
 * Bitmap used to describe if data follows a packet in either direction. Both
 * may be set, indicating an RPC has data accompanying a packet following both
 * the initial request and the subsequent response.
 */
typedef enum
{
   TO_HOST     = 0x1,
   FROM_HOST   = 0x2
} data_direction;

typedef struct
{
   size_t to_host;
   size_t from_host;
} payload_size;

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
		payload_size *size)             //! how much data
{
	bool has_payload = true;
	memset(size, 0, sizeof(*size));

	/* XXX XXX XXX The code in this function GREATLY depends on how the
	 * functions in the interposer are coded, as we look at specific arguments
	 * for meaningful data. It'd be cleaner to have some nice method for
	 * specifying payload parameters and size in the packet generically, instead
	 * of this damn jump table.
	 */
	switch (pkt->method_id) {
		case CUDA_GET_DEVICE_PROPERTIES:
			*direction = TO_HOST;
			size->to_host = sizeof(struct cudaDeviceProp);
			break;
		case CUDA_SETUP_ARGUMENT:
			*direction = FROM_HOST;
			size->from_host = pkt->args[1].arr_argi[0];
			break;
		case CUDA_MEMCPY_H2D:
			*direction = FROM_HOST;
			size->from_host = pkt->args[2].arr_argi[0];
			break;
		case CUDA_MEMCPY_D2H:
			*direction = TO_HOST;
			size->to_host = pkt->args[2].arr_argi[0];
			break;
		case CUDA_MEMCPY_TO_SYMBOL_H2D:
			*direction = FROM_HOST;
			size->from_host = pkt->args[2].arr_argi[0];
			break;
		case CUDA_MEMCPY_FROM_SYMBOL_D2H:
			*direction = TO_HOST;
			size->to_host = pkt->args[2].arr_argi[0];
			break;
		case  __CUDA_REGISTER_FAT_BINARY:
			*direction = FROM_HOST;
			size->from_host = pkt->args[1].argll;
			break;
		case __CUDA_REGISTER_FUNCTION:
			*direction = FROM_HOST;
			size->from_host = pkt->args[1].arr_argi[0];
			break;
		case __CUDA_REGISTER_VARIABLE:
			*direction = FROM_HOST;
			size->from_host = pkt->args[1].arr_argi[0];
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
		struct remote_cubin *rcubin)	//! CUBIN state necessary for symbol lookup
{
	int err, retval = 0;
	struct cuda_packet *pkt = payload->buffer; //! pkt placed at top of region

	BUG(!pkt);

	bool has_payload; //! any data an RPC requires is stored after the pkt
	data_direction direction;
	payload_size data_size;

	// Transactions always start with the packet itself. This allows both ends
	// to determine the remainder of the protocol from the method_id directly.
	err = conn_get(conn, pkt, sizeof(*pkt));
	if (err < 0) return -ENETDOWN;

	has_payload = cudarpc_has_payload(pkt, &direction, &data_size);

	if (has_payload && (direction & FROM_HOST)) {
		// TODO realloc buffer if size greater
		err = conn_get(conn, (pkt + 1), data_size.from_host);
		if (err < 0) return -ENETDOWN;
	}

	err = demux(pkt, &(rcubin->cubins));
	if (err < 0) return -1;

	// Always return the packet. Some RPCs don't need anything else.
	err = conn_put(conn, pkt, sizeof(*pkt));
	if (err < 0) return -ENETDOWN;

	if (has_payload && (direction & TO_HOST)) {
		err = conn_put(conn, (pkt + 1), data_size.to_host);
		if (err < 0) return -ENETDOWN;
	}

	// Update CUBIN registration counts. No need to lock as the assembly module
	// ensures only one hidden call is sent to each node an assembly maps to
	// (thus only one minion thread will update these).
	if (unlikely(pkt->method_id == __CUDA_REGISTER_FAT_BINARY)) {
		rcubin->reg_count++;
	} else if (unlikely(pkt->method_id == __CUDA_UNREGISTER_FAT_BINARY)) {
		rcubin->reg_count--;
		if (rcubin->reg_count <= 0)
			retval = -ECANCELED; // inform caller the CUDA call stream ended
	}

	return retval;
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

	wait_for_termination();

	teardown(); // doesn't return
}

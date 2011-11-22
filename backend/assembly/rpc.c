/**
 * TODO
 */

// System includes
#include <errno.h>
#include <stdbool.h>
#include <string.h>

// Project includes
#include <assembly.h>
#include <debug.h>
#include <util/list.h>

// Directory-immediate includes
#include "types.h"
#include "rpc.h"

/*-------------------------------------- DEFINITIONS -------------------------*/

//! Buffer size used to send and receive RPC messages.
#define RPC_BUFFER_SIZE		(sizeof(struct rpc_msg))

//! Port used to send/receive RPC messages. There is only one port because there
//! is only one MAIN node to listen on it
#define RPC_PORT			34865
#define RPC_PORT_STR		"34865"

enum rpc_action
{
	ASSEMBLY_JOIN = 1,	//! New node has joined assembly network
	ASSEMBLY_REQUEST,	//! Node requests new assembly
	ASSEMBLY_TEARDOWN,	//! Node tearing down assembly
	ASSEMBLY_LEAVE		//! Node is leaving assembly network
};

// Message state for a join
struct rpc_join
{
	struct node_participant participant;
};

// Message state for a request
struct rpc_request
{
	union {
		struct assembly_hint hint;	// sent to MAIN
		struct assembly assembly;		// response from MAIN
	} u;
};

// Message state for a teardown
struct rpc_teardown
{
	asmid_t asmid;
};

// Message state for a leave
struct rpc_leave
{
	char hostname[HOST_LEN];
};

//! The actual message format used to send RPCs to/from the MAIN node.
struct rpc_msg
{
	enum rpc_action action;
	union {
		struct rpc_join join;
		struct rpc_request request;
		struct rpc_teardown teardown;
		struct rpc_leave leave; // TODO pull hostname from socket directly
	} u;
	int status; //! Error code of the RPC
};

/**
 * State associated with an RPC connection, either the incoming connections MAIN
 * nodes associate with, or the single connection MINION nodes have with the
 * MAIN node.
 */
struct rpc_thread
{
	struct list_head link; //! used only for participant nodes (not the admin)
	bool is_alive;
	pthread_t tid;
	int exit_code;
	struct rpc_connection rpc_conn;
	char hostname[HOST_LEN]; //! host associated with this thread
};

/*
 * Functions we call in assembly.c to carry out execution of RPCs.
 */

extern struct assembly * assembly_find(asmid_t id);
extern int add_participant(struct node_participant *p);
extern int rm_participant(const char *hostname,
		struct node_participant **removed);
extern bool participant_exists(const char *hostname);

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct rpc_thread *admin_thread = NULL;

//! List of rpc_thread representing minions connected to us. A "minion" is a
//! thread that is assigned to a remote host. Multiple joins from the same
//! remote host are not allowed.
static struct list_head minions = LIST_HEAD_INIT(minions);
static pthread_mutex_t minion_lock = PTHREAD_MUTEX_INITIALIZER;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

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

/*-------------------------------------- THREADING FUNCTIONS -----------------*/

static void
minion_cleanup(void *arg)
{
	int err;
	struct rpc_thread *state = (struct rpc_thread*)arg;
	struct node_participant *node = NULL;

	state->is_alive = false;

	pthread_mutex_lock(&minion_lock);
	__rm_minion(state);
	pthread_mutex_unlock(&minion_lock);

#define EXIT_STRING	"rpc minion exit: "
	switch (state->exit_code) {
		case 0:
			printd(DBG_INFO, EXIT_STRING "success\n");
			break;
		case -ENETDOWN:
			printd(DBG_INFO, EXIT_STRING "error: network down\n");
			break;
		case -EHOSTDOWN:
			printf("Host %s disconnecting\n",
					state->hostname);
			break;
		default:
			BUG(1);
			break;
	}

	// FIXME If a host tries to join more than once, we should not lookup and remove
	// the participant state unless no minion rpc thread exists which is
	// assigned to this hostname already.

	if (participant_exists(state->hostname)) {
		printd(DBG_WARNING, EXIT_STRING "Host %s disconnecting uncleanly\n",
				state->hostname);
		err = rm_participant(state->hostname, &node);
		BUG(err < 0);
	}

	// XXX TODO Clean up remaining assembly state, too. This is vital.

	err = rpc_close(&state->rpc_conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: close rpc conn\n");
	}
	err = rpc_tini_conn(&state->rpc_conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: tini rpc conn\n");
	}
#undef EXIT_STRING
}

static void
do_msg(struct rpc_msg *msg)
{
	int err;
	// JOIN: message return status is -ENOMEM, zero or whatever add_participant
	// returns
	if (msg->action == ASSEMBLY_JOIN) {
		struct node_participant *new_node = malloc(sizeof(*new_node));
		if (!new_node) {
			fprintf(stderr, "Out of memory\n");
			printd(DBG_ERROR, "Out of memory\n");
			msg->status = -ENOMEM;
			return;
		}
		*new_node = msg->u.join.participant; // full structure copy
		INIT_LIST_HEAD(&new_node->link);
		err = add_participant(new_node);
		if (err < 0) { // participant could not be added, free it up
			printd(DBG_ERROR, "Host %s requesting join more than once\n",
					new_node->hostname);
			free(new_node);
		} else {
			printf("Host %s joined the assembly network\n",
					new_node->hostname);
		}
		msg->status = err;
	}
	
	else if (msg->action == ASSEMBLY_REQUEST) {
		struct assembly_hint *hint = &msg->u.request.u.hint;
		struct assembly *assm = NULL;
		asmid_t asmid = assembly_request(hint);
		if (!(VALID_ASSEMBLY_ID(asmid))) {
			msg->status = -1;
		} else {
			msg->status = 0;
			assm = assembly_find(asmid);
			msg->u.request.u.assembly = *assm;
		}
	}
	
	else if (msg->action == ASSEMBLY_TEARDOWN) {
		msg->status = assembly_teardown(msg->u.teardown.asmid);
	}

	else if (msg->action == ASSEMBLY_LEAVE) {
		struct node_participant *node = NULL;
		msg->status = rm_participant(msg->u.leave.hostname, &node);
		if (msg->status == 0) { // participant was removed, free memory
			free(node);
		} else {
			// Bug or protocol error... TODO Get rid of this printd
			printd(DBG_ERROR, "Could not remove participant %s\n",
					msg->u.leave.hostname);
		}
	}

	else {
		// No idea what message this is. An idiot or hacker caused this, or rays
		// from the sun. If an idiot, it's most likely me (or you).
		msg->status = -EPROTO;
	}
}

// Thread assigned to each participant node in the network to process its
// assembly RPC requests.
static void *
minion_thread(void *arg)
{
	int err;
	int old_thread_state;
	struct rpc_thread *state = (struct rpc_thread*)arg;
	struct sockconn *conn = &state->rpc_conn.sockconn;

	pthread_cleanup_push(minion_cleanup, state);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	state->is_alive = true;

	// TODO Get the hostname from the socket instead of snooping the message
	// action, as indicated below. hostname is required to remove state when the
	// remote host departs.

	while (1) {
		struct rpc_msg *msg;
		err = conn_get(conn, state->rpc_conn.buffer, RPC_BUFFER_SIZE);
		if (err < 0) { // always an error
			BUG(err == -EINVAL);
			state->exit_code = -ENETDOWN;
			break;
		}
		if (err == 0) { // remote socket was closed, node has left (expected)
			state->exit_code = -EHOSTDOWN;
			break;
		}
		msg = (struct rpc_msg*) state->rpc_conn.buffer;
		if (msg->action == ASSEMBLY_JOIN) // TODO I don't like putting this here
			strncpy(state->hostname, msg->u.join.participant.hostname, HOST_LEN);
		do_msg(msg);
		err = conn_put(&state->rpc_conn.sockconn,
				state->rpc_conn.buffer, RPC_BUFFER_SIZE);
		if (err < 0) { // always an error
			BUG(err == -EINVAL);
			state->exit_code = -ENETDOWN; // some other bad error
		}
		if (err == 0) { // remote socket was closed, node has left (expected)
			state->exit_code = -EHOSTDOWN;
			break;
		}
		memset(state->rpc_conn.buffer, 0, RPC_BUFFER_SIZE);
	}
	pthread_cleanup_pop(1); // invoke cleanup routine
	pthread_exit(NULL); // not reachable
}

static void
admission_cleanup(void *arg)
{
	int err;
	struct rpc_thread *state = (struct rpc_thread*)arg;

	state->is_alive = false;

#define EXIT_STRING	"rpc admin exit: "
	switch (state->exit_code) {
		case 0:
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

	err = rpc_close(&state->rpc_conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: close rpc conn\n");
	}
	err = rpc_tini_conn(&state->rpc_conn);
	if (err < 0) {
		printd(DBG_ERROR, EXIT_STRING "error: tini rpc conn\n");
	}
#undef EXIT_STRING

	// FIXME Check to see if we have other minion threads. Keep in mind
	// halt_rpc_thread does a join, but the admission thread detached them.

	free(admin_thread);
}

// this thread enables remote machines to join the assembly network by spawning
// minion threads to interact 1:1 with each remote host
static void *
admission_thread(void *arg)
{
	int err;
	int old_thread_state;
	struct rpc_thread *state = (struct rpc_thread*)arg;
	struct sockconn *conn = &state->rpc_conn.sockconn;

	pthread_cleanup_push(admission_cleanup, state);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_thread_state);
	state->is_alive = true;

	// Enable other nodes to connect to us
	err = conn_localbind(conn, RPC_PORT_STR);
	if (err < 0) {
		state->exit_code = -ENETDOWN;
		pthread_exit(NULL);
	}

	// TODO Print out or control which interfaces to listen on. I think
	// conn_localbind listens on the given port across all interfaces.

	// Listen for new connections. For each one, create a new thread.
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
		// FIXME fill in remaining rpc_connection fields
		err = rpc_init_conn(&new_state->rpc_conn);
		if (err < 0) {
			state->exit_code = err;
			break;
		}
		new_state->rpc_conn.sockconn = new_conn; // copy sock state
		err = pthread_create(&new_state->tid, NULL, minion_thread, new_state);
		if (err < 0) {
			state->exit_code = -EPROTO;
			break;
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
	pthread_exit(NULL); // not reachable
}

static inline void
halt_rpc_thread(struct rpc_thread *state)
{
	if (state->is_alive) {
		pthread_cancel(state->tid);
		pthread_join(state->tid, NULL);
	}
}

//! Examine return code from conn_put and conn_get. Codes are the same.
#define RPC_CHECK_TRANSACTION(err)					\
	do {											\
		if (err <= 0) {								\
			BUG(err == -EINVAL);					\
			if (err == 0) { /* main node died */	\
				exit_errno = -EHOSTDOWN;			\
			} else {								\
				exit_errno = -ENETDOWN;				\
			}										\
			goto fail;								\
		}											\
	} while (0)


//! Sends a message and receives a single reply.
static int
__send_msg(struct rpc_connection *conn, struct rpc_msg *msg)
{
	int err, exit_errno;

	BUG(!conn || !msg);

	err = conn_put(&conn->sockconn, msg, RPC_BUFFER_SIZE);
	RPC_CHECK_TRANSACTION(err);

	err = conn_get(&conn->sockconn, msg, RPC_BUFFER_SIZE);
	RPC_CHECK_TRANSACTION(err);

	return 0;

fail:
	return exit_errno;
}

/*-------------------------------------- MINION FUNCTIONS --------------------*/

int rpc_init_conn(struct rpc_connection *conn)
{
	int exit_errno;
	memset(conn, 0, sizeof(*conn));
	conn->buffer = calloc(1, RPC_BUFFER_SIZE);
	if (!conn->buffer) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	return 0;
fail:
	if (conn->buffer)
		free(conn->buffer);
	return exit_errno;
}

int rpc_tini_conn(struct rpc_connection *conn)
{
	free(conn->buffer);
	memset(conn, 0, sizeof(*conn));
	return 0;
}

int rpc_connect(struct rpc_connection *conn, const char *ip)
{
	int err, exit_errno;
	err = conn_connect(&conn->sockconn, ip, RPC_PORT_STR);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

int rpc_close(struct rpc_connection *conn)
{
	int err;
	if (!conn)
		goto fail;
	err = conn_close(&conn->sockconn);
	if (err < 0)
		goto fail;
	return 0;
fail:
	return -1;
}

int rpc_send_join(struct rpc_connection *conn, struct node_participant *p)
{
	int err, exit_errno;
	struct rpc_msg *msg = (struct rpc_msg*) conn->buffer;

	BUG(!msg);

	memset(msg, 0, sizeof(*msg));
	msg->action = ASSEMBLY_JOIN;
	msg->u.join.participant = *p; // whole struct copy

	err = __send_msg(conn, msg);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	return (msg->status);
fail:
	return exit_errno;
}

int rpc_send_leave(struct rpc_connection *conn)
{
	int err, exit_errno;
	struct rpc_msg *msg = (struct rpc_msg*) conn->buffer;

	BUG(!msg);

	memset(msg, 0, sizeof(*msg));
	msg->action = ASSEMBLY_LEAVE;
	err = gethostname(msg->u.leave.hostname, HOST_LEN);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}

	err = __send_msg(conn, msg);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	return (msg->status);
fail:
	return exit_errno;
}

int rpc_send_request(struct rpc_connection *conn,
		const struct assembly_hint *hint,
		struct assembly *assm)
{
	int err, exit_errno;
	struct rpc_msg *msg = (struct rpc_msg*) conn->buffer;

	BUG(!msg);

	memset(msg, 0, sizeof(*msg));
	msg->action = ASSEMBLY_REQUEST;
	msg->u.request.u.hint = *hint; // whole struct copy

	err = __send_msg(conn, msg);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	*assm = msg->u.request.u.assembly; // whole struct copy
	return (msg->status);
fail:
	return exit_errno;
}

int rpc_send_teardown(struct rpc_connection *conn, asmid_t asmid)
{
	int err, exit_errno;
	struct rpc_msg *msg = (struct rpc_msg*) conn->buffer;

	BUG(!msg);

	memset(msg, 0, sizeof(*msg));
	msg->action = ASSEMBLY_TEARDOWN;
	msg->u.teardown.asmid = asmid;

	err = __send_msg(conn, msg);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	return (msg->status);
fail:
	return exit_errno;
}

/*-------------------------------------- MAIN FUNCTIONS ----------------------*/

int rpc_enable(void)
{
	int err, exit_errno;
	if (admin_thread) {
		exit_errno = -EEXIST;
		goto fail;
	}
	admin_thread = calloc(1, sizeof(*admin_thread));
	if (!admin_thread) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	err = rpc_init_conn(&admin_thread->rpc_conn);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	err = pthread_create(&admin_thread->tid, NULL,
			admission_thread, (void*)admin_thread);
	if (err < 0) {
		// errno can only be EAGAIN, EPERM, or EINVAL
		exit_errno = -EPROTO;
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

void rpc_disable(void)
{
	if (admin_thread && admin_thread->is_alive)
		halt_rpc_thread(admin_thread);
	// the admin thread will free its conn
}

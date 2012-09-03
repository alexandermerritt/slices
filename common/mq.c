#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <debug.h>

#include <mq.h>

#define MQ_PERMS				(S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | \
														S_IROTH | S_IWOTH)
#define MQ_OPEN_OWNER_FLAGS		(O_RDONLY | O_CREAT | O_EXCL | O_NONBLOCK)
#define MQ_OPEN_CONNECT_FLAGS	(O_WRONLY)

#define MQ_ID_INVALID_VALUE		((mqd_t) - 1) // RTFM
#define MQ_ID_IS_VALID(m)		((m) != MQ_ID_INVALID_VALUE)

#define MQ_MAX_MESSAGES			8

//! Maximum message size allowable. Just set to the size of our structure.
#define MQ_MAX_MSG_SIZE			(sizeof(struct message))

//! Default priority for messages
#define MQ_DFT_PRIO				0

static struct mq_state daemon_mq;

struct message
{
    msg_event type;
    union {
        pid_t pid; /* to daemon ATTACH_CONNECT */
        bool allow; /* to pid ATTACH_CONNECT_ALLOW */
        /* TODO to daemon ATTACH_REQUEST_ASSEMBLY */
        assembly_key_uuid key; /* to pid ATTACH_ASSIGN_ASSEMBLY */
    } m; /* actual message data */
};

/* forward declaration */
static int set_notify(struct mq_state*);

// Receive a message. If we are sent some signal while receiving, we retry the
// receive. If the in MQ is empty, we spin.
static int
recv_message(struct mq_state *state, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_receive(state->id, (char*)msg, sizeof(*msg), NULL);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
        if (errno == EAGAIN)
            goto again; // MQ is empty, try again
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

static int
open_other_mq(struct mq_state *state)
{
    sprintf(state->name, "%s%d", ATTACH_NAME_PREFIX, state->pid);
    printd(DBG_INFO, "mq name '%s'\n", state->name);
    state->id = mq_open(state->name, MQ_OPEN_CONNECT_FLAGS);
    if (!MQ_ID_IS_VALID(state->id)) {
        perror("mq_open on incoming process");
        return -1;
    }
    return 0;
}

/* called by the mqueue layer on incoming messages */
static void
process_messages(struct mq_state *state)
{
    int err;
    struct message msg;

    if (!state || !state->valid)
        return;

    BUG(!MQ_ID_IS_VALID(state->id));

	// Re-enable notification on the message queue. The man page says
	// notifications are one-shot, and need to be reset after each trigger. It
	// additionally says that notifications are only triggered upon receipt of a
	// message on an EMPTY queue. Thus we set notification first, then empty the
	// queue completely.
    err = set_notify(state);
    if (err < 0) {
		fprintf(stderr, "Error setting notify on PID %d: %s\n",
				state->pid, strerror(-(err)));

		return;
	}

    while ( 1 ) {
        err = recv_message(state, &msg);
        if ( err < 0 ) {
            if ( err == -EAGAIN )
                break;
            fprintf(stderr, "Error recv msg on id %d: %s\n",
                    state->id, strerror(-(err)));
        }

        state->notify(msg.type, msg.m.pid);
    }
}

static void
__process_messages(union sigval sval)
{
    struct mq_state *state;
    /* Extract our specific state from the argument */
    state = (struct mq_state*) sval.sival_ptr;
    process_messages(state);
}

static int
send_message(struct mq_state *state, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_send(state->id, (char*)msg, sizeof(*msg), MQ_DFT_PRIO);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
		if (errno == EAGAIN)
			goto again; // MQ is full, try again (spin)
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

static inline int
set_notify(struct mq_state *state)
{
	struct sigevent event;
	memset(&event, 0, sizeof(event));
	event.sigev_notify = SIGEV_THREAD;
	event.sigev_notify_function = __process_messages;
	event.sigev_value.sival_ptr = state;
	if ( 0 > mq_notify(state->id, &event) )
    {
        perror("mq_notify");
        return -1;
    }
	return 0;
}

/*
 * daemon functions
 */

int attach_open(msg_recv_callback notify)
{
    struct mq_attr qattr;

    if ( !notify )
        return -1;

    memset(&daemon_mq, 0, sizeof(daemon_mq));
    daemon_mq.notify = notify;
    snprintf(daemon_mq.name, MAX_LEN, "%s", ATTACH_DAEMON_MQ_NAME);
    daemon_mq.pid = -1; /* not used for open() */
    daemon_mq.valid = true;

    memset(&qattr, 0, sizeof(qattr));
    qattr.mq_maxmsg = MQ_MAX_MESSAGES;
    qattr.mq_msgsize = MQ_MAX_MSG_SIZE;
    daemon_mq.id =
        mq_open(daemon_mq.name, MQ_OPEN_OWNER_FLAGS, MQ_PERMS, &qattr);
    if ( !MQ_ID_IS_VALID(daemon_mq.id) ) {
        perror("mq_open");
        if ( errno == EEXIST )
            fprintf(stderr, "> Daemon already running in another instance,"
                    " or previously crashed and old MQ was not cleaned up\n");
        return -1;
    }

    set_notify(&daemon_mq);
    return 0;
}

int attach_close(void)
{
    daemon_mq.valid = false;
    if (0 > mq_close(daemon_mq.id)) {
        perror("mq_close");
        return -1;
    }
    daemon_mq.id = MQ_ID_INVALID_VALUE;
    if (0 > mq_unlink(daemon_mq.name)) {
        perror("mq_unlink");
        return -1;
    }
    return 0;
}

int attach_allow(struct mq_state *state, pid_t pid)
{
    if ( !state ) return -1;
    state->pid = pid;
    return open_other_mq(state);
}

int attach_send_allow(struct mq_state *state, bool allow)
{
    struct message msg;
    msg.type = ATTACH_CONNECT_ALLOW;
    msg.m.allow = allow;
    if (0 > send_message(state, &msg)) {
        fprintf(stderr, "Error sending message to PID %d\n", state->pid);
        return -1;
    }
    return 0;
}

int attach_send_assembly(struct mq_state *state, assembly_key_uuid key)
{
    struct message msg;
    msg.type = ATTACH_ASSIGN_ASSEMBLY;
    memcpy(msg.m.key, key, sizeof(assembly_key_uuid));
    if (0 > send_message(state, &msg)) {
        fprintf(stderr, "Error sending message to PID %d\n", state->pid);
        return -1;
    }
    return 0;
}

/*
 * Interposer functions
 */

int attach_init(struct mq_state *recv, struct mq_state *send)
{
	struct mq_attr qattr;

    if (!recv || !send)
        return -1;

    memset(&qattr, 0, sizeof(qattr));
	qattr.mq_maxmsg = MQ_MAX_MESSAGES;
	qattr.mq_msgsize = MQ_MAX_MSG_SIZE;

    recv->notify = NULL; /* interposer recvs synchronously */
    recv->pid = -1; /* only daemon code uses this field */
    snprintf(recv->name, MAX_LEN, "%s%d", ATTACH_NAME_PREFIX, getpid());
    recv->id = mq_open(recv->name, MQ_OPEN_OWNER_FLAGS, MQ_PERMS, &qattr);
    if (!MQ_ID_IS_VALID(recv->id)) {
        perror("mq_open");
        return -1;
    }

    snprintf(send->name, MAX_LEN, "%s", ATTACH_DAEMON_MQ_NAME);
    send->notify = NULL; /* send is an outbound mq */
    send->pid = -1;
    send->id = mq_open(send->name, MQ_OPEN_CONNECT_FLAGS, MQ_PERMS, qattr);
    if (!MQ_ID_IS_VALID(send->id)) {
        perror("mq_open");
        return -1;
    }

    return 0;
}

int attach_tini(struct mq_state *recv, struct mq_state *send)
{
    if (0 > mq_close(recv->id)) {
        perror("mq_close");
        return -1;
    }
    if (0 > mq_unlink(recv->name)) {
        perror("mq_unlink");
        return -1;
    }

    if (0 > mq_close(send->id)) {
        perror("mq_close");
        return -1;
    }

    return 0;
}

int attach_send_connect(struct mq_state *recv, struct mq_state *send)
{
    struct message msg;

    if (!send)
        return -1;

    msg.type = ATTACH_CONNECT;
    msg.m.pid = getpid();
    if (0 > send_message(send, &msg)) {
        fprintf(stderr, "Error sending message to daemon\n");
        return -1;
    }

    /* block until daemon sends the okay */
    if (0 > recv_message(recv, &msg)) {
        fprintf(stderr, "Error receving message from daemon\n");
        return -1;
    }
    if (msg.type == ATTACH_CONNECT_ALLOW) {
        if (!msg.m.allow) {
            fprintf(stderr, "Not allowed to connect to daemon\n");
            return -1;
        }
    } else {
        fprintf(stderr, "Unexpected message recieved: %d\n", msg.type);
        return -1;
    }
    return 0;
}

int attach_send_disconnect(struct mq_state *recv, struct mq_state *send)
{
    struct message msg;

    if (!send)
        return -1;

    msg.type = ATTACH_DISCONNECT;
    msg.m.pid = getpid();
    if (0 > send_message(send, &msg)) {
        fprintf(stderr, "Error sending message to daemon\n");
        return -1;
    }

    /* no response expected */

    return 0;
}

/* TODO convey assembly configuration file somewhere */
int attach_send_request(struct mq_state *recv, struct mq_state *send,
        assembly_key_uuid key)
{
    struct message msg;

    if (!recv || !send)
        return -1;

    msg.type = ATTACH_REQUEST_ASSEMBLY;
    msg.m.pid = getpid(); /* so the runtime knows which return queue to use */
    if (0 > send_message(send, &msg)) {
        fprintf(stderr, "Error sending message to daemon\n");
        return -1;
    }

    /* block until daemon has exported assembly for us */
    if (0 > recv_message(recv, &msg)) {
        fprintf(stderr, "Error receving message from daemon\n");
        return -1;
    }
    if (msg.type == ATTACH_ASSIGN_ASSEMBLY) {
        memcpy(key, msg.m.key, sizeof(assembly_key_uuid));
    } else {
        fprintf(stderr, "Unexpected message recieved: %d\n", msg.type);
        return -1;
    }
    return 0;
}

void attach_cleanup(void)
{
    /* TODO figure out how to get a list of all mqueues that exist, match the
     * name to the prefix we have and remove. */
}

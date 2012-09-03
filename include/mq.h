/**
 * @file attach.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2012-08-25
 * @brief Interface for the interposer library to communicate with the shadowfax
 * daemon.
 */

#ifndef _ATTACH_H
#define _ATTACH_H

#include <stdbool.h>
#include <unistd.h>
#include <mqueue.h>
#include <assembly.h>

/* mq names must start with / */
#define ATTACH_NAME_PREFIX      "/shadowfax_mq_"
#define ATTACH_DAEMON_MQ_NAME   ATTACH_NAME_PREFIX "daemon"

#define MAX_LEN				255

/* callback definitions */
typedef enum
{
    /* daemon recv events */
    ATTACH_CONNECT = 1,
    ATTACH_DISCONNECT,
    ATTACH_CONNECT_ALLOW,
    ATTACH_REQUEST_ASSEMBLY,
    /* interposer recv events */
    ATTACH_ASSIGN_ASSEMBLY
} msg_event;

struct mq_state; /* forward declaration */
typedef void (*msg_recv_callback)(msg_event e, pid_t pid);

/* connection state */
struct mq_state
{
    bool valid;
    char name[MAX_LEN];
    pid_t pid;
    mqd_t id;
    msg_recv_callback notify;
};

/* daemon functions */
int attach_open(msg_recv_callback notify);
int attach_close(void);
int attach_allow(struct mq_state *state, pid_t pid);
int attach_send_allow(struct mq_state *state, bool allow);
int attach_send_assembly(struct mq_state *state, assembly_key_uuid key);

/* interposer functions */
/* interposer does not receive asynchronous message notification from daemon */
int attach_init(struct mq_state *recv, struct mq_state *send);
int attach_tini(struct mq_state *recv, struct mq_state *send);
int attach_send_connect(struct mq_state *recv, struct mq_state *send);
int attach_send_disconnect(struct mq_state *recv, struct mq_state *send);
int attach_send_request(struct mq_state *recv, struct mq_state *send,
        assembly_key_uuid key);

/* if code crashes, call this to remove files this interface may have created
 * which were not cleaned up */
void attach_cleanup(void);

#endif /* _ATTACH_H */

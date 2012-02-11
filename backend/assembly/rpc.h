/**
 * TODO
 *
 * The macro RPC_USE_HATI enables or disables use of the HATI library. Don't
 * enable it just yet, as the code has been written to use sockets for now.
 */

#ifndef _RPC_H
#define _RPC_H

// System includes
#include <stdbool.h>

// Project includes
#include <io/sock.h>

// Directory-immediate includes
#include "types.h"

/**
 * Encapsulated conn associated with an RPC connection.
 */
struct rpc_connection
{
	//char ip[RPC_MAX_IP_LEN];//! Address of node to connect to, if outbound
	//int port;				//! Connect port (outbound) or listen port (inbound)
	struct sockconn sockconn;
	void *buffer;			//! Buffer containing RPC messages
};

/*-------------------------------------- MINION FUNCTIONS --------------------*/

int rpc_connect(struct rpc_connection *conn, const char *main_ip);

// functions analogous to those in the assembly API which the MAIN node must see
int rpc_send_join(struct rpc_connection *conn, struct node_participant *p);
int rpc_send_request(struct rpc_connection *conn,
		const struct assembly_hint *hint,
		struct assembly *assm);
int rpc_send_teardown(struct rpc_connection *conn, asmid_t asmid);
int rpc_send_leave(struct rpc_connection *conn);

/*-------------------------------------- MAIN FUNCTIONS ----------------------*/

// The existance of the rpc thread either allows or prevents minions from
// joining and having assemblies mapped to them (i.e. minions won't be able to
// add their node_participant data to the list in MAIN).
int rpc_enable(void);
void rpc_disable(void);

/*-------------------------------------- COMMON FUNCTIONS --------------------*/

int rpc_init_conn(struct rpc_connection *conn);
int rpc_tini_conn(struct rpc_connection *conn);
int rpc_close(struct rpc_connection *conn);

#endif	/* _RPC_H */

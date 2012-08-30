#ifndef _INTERNALS_H
#define _INTERNALS_H

// System includes
#include <errno.h>
#include <ifaddrs.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// Project includes
#include <assembly.h>
#include <cuda/fatcubininfo.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <debug.h>
#include <util/list.h>
#include <util/timer.h>

// Directory-immediate includes
#include "remote.h"
#include "rpc.h"
#include "types.h"

struct main_state
{
	//! Used by main node to assign global assembly IDs
	asmid_t next_asmid;

	//! List of node_participant structures representing available nodes.
	struct list_head participants;
	pthread_mutex_t plock; //! participant list lock
	
	// RPC thread state is contained inside rpc.c
	// Functions exist to determine if the thread is alive or not
};

struct minion_state
{
	//! State associated with an RPC connection to the MAIN node
	struct rpc_connection rpc_conn;

	// We do not maintain a participant list, as that list's sole purpose is for
	// assisting in creating assemblies, which the main node is responsible for.
};

/**
 * Assembly module internal state. Describes node configuration and contains set
 * of assemblies created. The assembly list within the main node will contain
 * ALL assemblies through the network. Within minions it will only be populated
 * with assemblies for which the minion directly has requested from the main
 * node (via local applications starting up). Mappers are merely tools and don't
 * do anything logistically (tool=one who lacks the mental capacity to know he
 * is being used).
 */
struct internals_state
{
	enum node_type type;
	pthread_mutex_t lock; //! Lock for changes to this struct
	struct list_head assembly_list;
	union node_specific {
		struct main_state main;
		struct minion_state minion;
		// no mapper state required
	} n;
	pid_t rsink_pid; //! PID of remote sink process
	int rsink_kill_sig; //! Signal used to terminate the remote sink process
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

extern struct internals_state *internals;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

#define NEXT_ASMID	(internals->n.main.next_asmid++)

#endif  /* _INTERNALS_H */

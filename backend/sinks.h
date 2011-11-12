/**
 * @file sinks.h
 * @date 2011-10-27
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief TODO
 *
 * TODO Clean up this file.
 */

#ifndef _SINKS_H
#define _SINKS_H

#include <assembly.h>

/*-------------------------------------- DEFINITIONS/CONFIGURATION -----------*/

//! Signal sinks should wait for from the parent runtime for termination.
#define SINK_TERM_SIG	SIGTERM

/*-------------------------------------- DATA TYPES --------------------------*/

/**
 * Structure to hold state pertaining to each sink forked from the runtime
 * process.
 */
struct sink_child
{
	struct list_head link;
	pid_t pid;
	enum {SINK_EXEC_LOCAL = 1, SINK_EXEC_REMOTE} type;
	asmid_t asmid;
};

/*-------------------------------------- COMMON SINK FUNCTIONS ---------------*/

void localsink(asmid_t asmid, pid_t pid);
int nv_exec_pkt(volatile struct cuda_packet *pkt);

/**
 * Fork a new sink process to associate with a newly created CUDA process that
 * has joined the assembly runtime.
 */
int fork_localsink(asmid_t asmid, pid_t memb_pid, pid_t *childpid);

#endif

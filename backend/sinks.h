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

//! Environment variable used to pass the UUID to the local sinks.
#define SINK_ASSM_EXPORT_KEY_ENV	"ASSM_KEY"

//! Environment variable used to pass the PID a local sink attaches to.
#define SINK_SHMGRP_PID				"SHMGRP_PID"

//! Max length of a VAR=VAL environment setting. -1 for the NUL byte
#define ENV_MAX_LEN		511

//! argv[0] of the exec'd sink, so we know it was created by us and not executed
//! on the command line (a crude method)
#define SINK_EXEC_NAME	"localsink-814227d8-a367-40ce-be9f-b965c8376369"

/*-------------------------------------- DATA TYPES --------------------------*/

/**
 * Structure to hold state pertaining to each sink forked from the runtime
 * process.
 */
struct sink
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

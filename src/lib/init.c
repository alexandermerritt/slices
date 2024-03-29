/**
 * @file: init.c
 * @author: Alexander Merritt, merritt.alex@gatech.edu
 * @desc library initialization code
 */

//===----------------------------------------------------------------------===//
// Includes
//===----------------------------------------------------------------------===//

// System includes
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

// CUDA includes
#include <__cudaFatFormat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>

// Project includes
#include <assembly.h>
#include <cuda/bypass.h>
#include <cuda/hidden.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <cuda/packet.h> 
#include <debug.h>
#include <mq.h>
#include <util/compiler.h>
#include <util/x86_system.h>

//===----------------------------------------------------------------------===//
// Forward declarations
//===----------------------------------------------------------------------===//

extern bool scheduler_joined;
extern struct mq_state recv_mq, send_mq;

extern asmid_t assm_id;
extern struct assembly_hint hint;

/* forward delcaration into assembly/cuda_interface.c */
extern int assm_cuda_init(asmid_t id);
extern int assm_cuda_tini(void);

//===----------------------------------------------------------------------===//
// Daemon-attachment functions
//===----------------------------------------------------------------------===//

int join_scheduler(void)
{
    int err;
    assembly_key assm_key;

    if (scheduler_joined)
        return 0;

    scheduler_joined = true;

    memset(&recv_mq, 0, sizeof(recv_mq));
    memset(&send_mq, 0, sizeof(send_mq));

    /* initialize interfaces */
    err = attach_init(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_init: %d\n", err);
        fprintf(stderr, "Error attach_init: %d\n", err);
        return -1;
    }
	err = assembly_runtime_init(NODE_TYPE_MAPPER, NULL);
    if (err < 0) {
        printd(DBG_ERROR, "Error initializing assembly state\n");
        fprintf(stderr, "Error initializing assembly state\n");
        return -1;
    }

    /* tell daemon we wish to join */
    err = attach_send_connect(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_send_connect: %d\n", err);
        fprintf(stderr, "Error attach_send_connect: %d\n", err);
        return -1;
    }

    /* read the hint then send it to the daemon */
    err = assembly_read_hint(&hint);
    if (err < 0) {
        fprintf(stderr, "> Error reading hint file\n");
        return -1;
    }
    err = attach_send_request(&recv_mq, &send_mq, &hint, &assm_key);
    if (err < 0) {
        printd(DBG_ERROR, "Error attach_send_request: %d\n", err);
        fprintf(stderr, "Error attach_send_request: %d\n", err);
        return -1;
    }

    /* 'fix' the assembly on the network */
    err = assembly_import(&assm_id, assm_key);
    BUG(assm_id == INVALID_ASSEMBLY_ID);
    if (err < 0) {
        printd(DBG_ERROR, "Error assembly_import: %d\n", err);
        fprintf(stderr, "Error assembly_import: %d\n", err);
        return -1;
    }
    assembly_print(assm_id);
    err = assembly_map(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error assembly_map: %d\n", err);
        fprintf(stderr, "Error assembly_map: %d\n", err);
        return -1;
    }

    /* XXX initialize the assembly/cuda_interface.c state AFTER the assembly
     * interface has been initialized AND an assembly has been imported */
    err = assm_cuda_init(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error initializing assembly cuda interface\n");
        fprintf(stderr, "Error initializing assembly cuda interface\n");
        return -1;
    }

    return 0;
}

int leave_scheduler(void)
{
    int err;

    if (!scheduler_joined)
        return -1;

    scheduler_joined = false;

    err = attach_send_disconnect(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error telling daemon disconnect\n");
        fprintf(stderr, "Error telling daemon disconnect\n");
        return -1;
    }

    /* remove state associated with the assembly cuda interface */
    err = assm_cuda_tini();
    if (err < 0) {
        printd(DBG_ERROR, "Error deinitializing assembly cuda interface\n");
        fprintf(stderr, "Error deinitializing assembly cuda interface\n");
        return -1;
    }

    err = assembly_teardown(assm_id);
    if (err < 0) {
        printd(DBG_ERROR, "Error destroying assembly\n");
        fprintf(stderr, "Error destroying assembly\n");
        return -1;
    }

    err = assembly_runtime_shutdown();
    if (err < 0) {
        printd(DBG_ERROR, "Error cleaning up assembly runtime\n");
        fprintf(stderr, "Error cleaning up assembly runtime\n");
        return -1;
    }

    err = attach_tini(&recv_mq, &send_mq);
    if (err < 0) {
        printd(DBG_ERROR, "Error cleaning up attach state\n");
        fprintf(stderr, "Error cleaning up attach state\n");
        return -1;
    }

    return 0;
}

//===----------------------------------------------------------------------===//
// Library constructors
//===----------------------------------------------------------------------===//

#if 0
__attribute__((constructor)) void sfinit(void)
{
    if (strstr(getenv("_"), "gdb"))
        return;
    if (scheduler_joined)
        return;
    fill_bypass(&bypass);
    if (join_scheduler())
        fprintf(stderr, ">> Error attaching to daemon\n");
    else
        printf(">> Attached to daemon.\n");
}

__attribute__((destructor)) void sftini(void)
{
    if (strstr(getenv("_"), "gdb"))
        return;
    if (!scheduler_joined)
        return;
    if (leave_scheduler())
        fprintf(stderr, ">> Error leaving daemon\n");
    else
        printf(">> Detached from daemon.\n");
}
#endif

/**
 * @file cuda_runtime.c
 * @date Feb 27, 2011
 * @author Alex Merritt, merritt.alex@gatech.edu
 */

#define _GNU_SOURCE

//===----------------------------------------------------------------------===//
// Library includes
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
#include <uuid/uuid.h>

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

// Directory-immediate includes
#include "timing.h"

#define USERMSG_PREFIX "=== INTERPOSER === "

//===----------------------------------------------------------------------===//
// Global variables (in glob.c)
//===----------------------------------------------------------------------===//

extern cudaError_t cuda_err;

extern unsigned int num_registered_cubins;

extern bool scheduler_joined;
extern struct mq_state recv_mq, send_mq;

extern asmid_t assm_id;
extern struct assembly_hint hint;

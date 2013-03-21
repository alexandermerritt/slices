#include "lib.h"

//===----------------------------------------------------------------------===//
// Global variables
//===----------------------------------------------------------------------===//

cudaError_t cuda_err = cudaSuccess;

unsigned int num_registered_cubins = 0;

bool scheduler_joined = false;
struct mq_state recv_mq, send_mq;

asmid_t assm_id;
struct assembly_hint hint;

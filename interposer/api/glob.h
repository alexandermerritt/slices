/**
 * @file glob.h
 * @author Alexander Merritt, merritt.alex@gatech.edu
 * @desc Assembly interposer globals
 */

#ifndef INTERPOSER_ASSM_GLOB_INCLUDED
#define INTERPOSER_ASSM_GLOB_INCLUDED

#include <cuda/bypass.h>
#include <assembly.h>
#include "internals.h"

/* Things to know
 * - thread/vgpu association; if none exist, must put thread into some vgpu
 * - if vgpu is remote or local
 * - where the marshaling region is, for remote vgpus
 */

/* association between an application thread and vgpu in the assembly */
/* assume one assembly is used for now */
struct tinfo
{
    bool valid;
    /* application state */
    pthread_t tid;
    /* vgpu state */
    struct vgpu_mapping *vgpu;
    /* marshaling state (only if remote) */
    void *buffer;
};

struct tinfo *__lookup(pthread_t tid);

#define rpc(tinfo)          ((tinfo)->vgpu->rpc)
#define VGPU_IS_LOCAL(vgpu) ((vgpu)->fixation == VGPU_LOCAL)

extern struct assembly * assembly_find(asmid_t id);

/*
 * Preprocessor magic to reduce typing
 */

/*
 * Function setup code; lookup thread-specific state (include in marshaling
 * time).
 */
#define FUNC_SETUP \
    void *buf = NULL; \
    struct tinfo *tinfo; \
    TIMER_DECLARE1(t); \
    TIMER_START(t); \
    tinfo = __lookup(pthread_self())
#define FUNC_SETUP_CERR \
    FUNC_SETUP; \
    cudaError_t cerr = cudaSuccess

/* initialize the buf ptr once thread state has been looked up */
static inline void
init_buf(void **buf, struct tinfo *tinfo)
{
    *buf = tinfo->buffer;
    memset(*buf, 0, sizeof(struct cuda_packet));
}


#endif

#include "glob.h"

//===----------------------------------------------------------------------===//
// Global variables (assembly-private)
//===----------------------------------------------------------------------===//

/* We assume only one assembly/process */
asmid_t assm_id;
struct assembly *assm;

int num_tids;
/* XXX limited to 32 threads per application process */
struct tinfo tinfos[32];
pthread_mutex_t tinfo_lock = PTHREAD_MUTEX_INITIALIZER;

//===----------------------------------------------------------------------===//
// Global functions (assembly-private)
//===----------------------------------------------------------------------===//

/* TODO need some way of specifying vgpu mapping */
/* TODO return vgpu_id */
/* TODO lock only if adding/removing entry, not when looking up */
/* TODO add only at first !valid slot, not keep appending at end */
struct tinfo *__lookup(pthread_t tid)
{
    int i;
    struct tinfo *ret = NULL;
    pthread_mutex_lock(&tinfo_lock);
    for (i = 0; i < num_tids; i++) {
        if (!tinfos[i].valid)
            continue;
        if (0 != pthread_equal(tinfos[i].tid, tid))
            break;
    }
    if (i < num_tids) { /* found tid state (more likely) */
        ret = &tinfos[i];
    } else { /* not found, make new entry */
        ret = &tinfos[num_tids++];
        ret->valid = true;
        ret->tid = tid;
        /* TODO only allocate if vgpu thread picks is remote */
        ret->buffer = malloc(1UL << 30); /* XXX hardcoded... */
        ret->vgpu = &assm->mappings[0]; /* default 0 until setDevice called */
        if (!ret->buffer) {
            fprintf(stderr, "Out of memory\n");
            abort();
        }
    }
    pthread_mutex_unlock(&tinfo_lock);
    BUG(!ret);
    return ret;
}


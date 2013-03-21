#include "glob.h"

/* XXX must be called before first cuda call is made
 * Interposer connects to runtime and queries for an assembly. runtime exports
 * it and provides it to the interposer. interposer then tells us what the
 * assembly ID is that we are supposed to use. */
int assm_cuda_init(asmid_t id)
{
    memset(tinfos, 0, sizeof(tinfos));
    assm_id = id;
    assm = assembly_find(assm_id);
    BUG(!assm);
    return 0;
}

/* should be called when no more cuda calls are made by application */
int assm_cuda_tini(void)
{
    int i;
    for (i = 0; i < num_tids; i++)
        if (tinfos[i].valid && tinfos[i].buffer)
            free(tinfos[i].buffer);
    memset(tinfos, 0, sizeof(tinfos));
    num_tids = 0;
    assm = NULL;
    return 0;
}

void cudaJmpTblConstructor(void)
{
}

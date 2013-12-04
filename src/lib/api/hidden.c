#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Hidden Registration
//===----------------------------------------------------------------------===//

static int num_binaries = 0;

int join_scheduler(void);
int leave_scheduler(void);
extern bool scheduler_joined;

/* TODO duplicate call */
void**
assm__cudaRegisterFatBinary(void *cubin)
{
    void *buf = NULL;
    struct tinfo *tinfo;
    void** ret;

    if (!scheduler_joined) {
        if (join_scheduler()) {
            fprintf(stderr, ">> Error attaching to daemon\n");
            exit(1);
        }
        fill_bypass(&bypass);
    }
    num_binaries++;

    BUG(!(tinfo = __lookup(pthread_self())));

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        ret = bypass.__cudaRegisterFatBinary(cubin);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFatBinary(buf, (buf + sizeof(struct cuda_packet)), cubin);
        rpc_ops.registerFatBinary(buf, NULL, rpc(tinfo));
        ret = cpkt_ret_hdl(buf);
    }
    printd(DEBUG_INFO, "ret %p\n", ret);
    return ret;
}

void assm__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFunction(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
                tid, bid, bDim, gDim, wSize);
        rpc_ops.registerFunction(buf, NULL, rpc(tinfo));
    }
}

extern void dump_flushes(void);
void assm__cudaUnregisterFatBinary(void** fatCubinHandle)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaUnregisterFatBinary(fatCubinHandle);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaUnregisterFatBinary(buf, fatCubinHandle);
        rpc_ops.unregisterFatBinary(buf, NULL, rpc(tinfo));
    }

    if (scheduler_joined && --num_binaries == 0)
        if (leave_scheduler())
            fprintf(stderr, ">> Error detaching to daemon\n");
    dump_flushes();
}

void assm__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char
        *deviceAddress, const char *deviceName, int ext, int vsize,
        int constant, int global)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
                deviceName, ext, vsize, constant, global);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterVar(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                vsize, constant, global);
        rpc_ops.registerVar(buf, NULL, rpc(tinfo));
    }
}

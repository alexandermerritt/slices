#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Hidden Registration
//===----------------------------------------------------------------------===//

void** assm__cudaRegisterFatBinary(void *cubin, struct rpc_latencies *lat)
{
    /* TODO duplicate call */
    FUNC_SETUP;
    void** ret;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        ret = bypass.__cudaRegisterFatBinary(cubin);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFatBinary(buf, (buf + sizeof(struct cuda_packet)), cubin);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.registerFatBinary(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        ret = cpkt_ret_hdl(buf);
        LAT_UPDATE(lat, buf);
    }
    return ret;
}

void assm__cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterFunction(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit,
                tid, bid, bDim, gDim, wSize);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.registerFunction(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

void assm__cudaUnregisterFatBinary(void** fatCubinHandle, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaUnregisterFatBinary(fatCubinHandle);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaUnregisterFatBinary(buf, fatCubinHandle);
        TIMER_END(t, lat->lib.setup);
        TIMER_START(t);
        rpc_ops.unregisterFatBinary(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

void assm__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char
        *deviceAddress, const char *deviceName, int ext, int vsize,
        int constant, int global, struct rpc_latencies *lat)
{
    FUNC_SETUP;

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        bypass.__cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
                deviceName, ext, vsize, constant, global);
        TIMER_END(t, lat->lib.wait);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaRegisterVar(buf, (buf + sizeof(struct cuda_packet)),
                fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                vsize, constant, global);
        TIMER_END(t, lat->lib.setup);
        rpc_ops.registerVar(buf, NULL, rpc(tinfo));
        TIMER_END(t, lat->lib.wait);
        LAT_UPDATE(lat, buf);
    }
}

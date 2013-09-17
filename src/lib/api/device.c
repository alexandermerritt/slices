#include "glob.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Device Management
//===----------------------------------------------------------------------===//

cudaError_t
assm_cudaGetDevice(int *device)
{
    struct tinfo *tinfo = __lookup(pthread_self());
    BUG(!tinfo);
    *device = tinfo->vgpu->vgpu_id;
    return cudaSuccess;
}

cudaError_t
assm_cudaGetDeviceCount(int *count)
{
    BUG(!assm);
    *count = assm->num_gpus;
    return cudaSuccess;
}

cudaError_t
assm_cudaGetDeviceProperties(struct cudaDeviceProp *prop,int device)
{
    FUNC_SETUP;
    init_buf(&buf, tinfo);
    if (device < 0 || device >= assm->num_gpus)
        return cudaErrorInvalidDevice;
    /* don't need to translate vgpu to pgpu ID since the index into mappings is
     * virtual IDs already */
    memcpy(prop, &assm->mappings[device].cudaDevProp, sizeof(*prop));
    printd(DBG_DEBUG, "name=%s\n", assm->mappings[device].cudaDevProp.name);
    return cudaSuccess;
}

cudaError_t
assm_cudaSetDevice(int device)
{
    FUNC_SETUP_CERR;
    if (device >= assm->num_gpus)
        return cudaErrorInvalidDevice;
    tinfo->vgpu = &assm->mappings[device];
    /* translate vgpu device ID to physical ID */
    device = tinfo->vgpu->pgpu_id;
    /* let it pass through so the driver makes the association */
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetDevice(device);
        BUG(bypass.cudaSetDeviceFlags(cudaDeviceBlockingSync) != cudaSuccess);
        //printf(">> set device flags to blocking sync\n");
    } else {
        init_buf(&buf, tinfo);
        /* XXX should we send a flushing call to clear the batch queue here? */
        pack_cudaSetDeviceFlags(buf, cudaDeviceBlockingSync);
        rpc_ops.setDeviceFlags(buf, NULL, rpc(tinfo));
        pack_cudaSetDevice(buf, device);
        rpc_ops.setDevice(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t
assm_cudaSetDeviceFlags(unsigned int flags)
{
    FUNC_SETUP_CERR;
    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetDeviceFlags(flags);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetDeviceFlags(buf, flags);
        rpc_ops.setDeviceFlags(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}

cudaError_t
assm_cudaSetValidDevices(int *device_arr, int len)
{
    FUNC_SETUP_CERR;

    /* XXX This function is ignored from within cuda_runtime.c */

    if (VGPU_IS_LOCAL(tinfo->vgpu)) {
        cerr = bypass.cudaSetValidDevices(device_arr, len);
    } else {
        init_buf(&buf, tinfo);
        pack_cudaSetValidDevices(buf, ((struct cuda_packet*)buf) + 1,
                device_arr, len);
        rpc_ops.setValidDevices(buf, NULL, rpc(tinfo));
        cerr = cpkt_ret_err(buf);
    }
    return cerr;
}


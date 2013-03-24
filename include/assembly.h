/**
 * @file assembly.h
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
 */
#ifndef _ASSEMBLY_H
#define _ASSEMBLY_H

// System includes
#include <uuid/uuid.h>
#include <string.h>

// Project includes
#include <cuda/packet.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

#define ASSEMBLY_SHMGRP_KEY "cudarpc"

#define MAX_GPUS_PER_NODE   8

/**
 * Distinguish between the centralized location where state of all assemblies is
 * located, or an instance on another node (minion) that forwards RPCs to the
 * main node to carry out the work. The third option is to initialize the
 * library to be a mapper, meaning it will import exported assemblies created by
 * a MAIN or MINION node for execution.
 */
enum node_type
{
    NODE_TYPE_UNINITIALIZED = 0,
    NODE_TYPE_MAIN,
    NODE_TYPE_MINION,
    NODE_TYPE_MAPPER,
    NODE_TYPE_INVALID   //! Used to compare against valid type values
};

#define INVALID_ASSEMBLY_ID     (0UL)
#define VALID_ASSEMBLY_ID(id)   ((id) != INVALID_ASSEMBLY_ID)

typedef unsigned long asmid_t;

enum hint_nic_type
{
    HINT_USE_ETH = 1,
    HINT_USE_IB // Can be IPoIB, or SDP
    // TODO HINT_USE_RDMA
};
#define HINT_ETH_STR    "eth"
#define HINT_IB_STR     "ib"

enum hint_policy
{
    HINT_ENUM_POLICY_INVALID = 0,
    HINT_ENUM_POLICY_LOCALFIRST,
    HINT_ENUM_POLICY_REMOTEONLY
};

#define HINT_COMMENT_CHAR               '#'
#define HINT_EQUAL_DELIM                "="
#define HINT_FIELD_MPI                  "mpi"
#define HINT_FIELD_POLICY               "policy"
    #define HINT_FIELD_POLICY_REMOTEONLY    "remote_only"
    #define HINT_FIELD_POLICY_LOCALFIRST    "local_first"
#define HINT_FIELD_BATCHSIZE            "batch_size"
#define HINT_FIELD_NIC                  "nic"
    #define HINT_FIELD_NIC_ETH          "eth"
    #define HINT_FIELD_NIC_SDP          "sdp"

/* set this env var to the full path of a hint file for it to be read */
#define HINT_ENV_VAR                    "ASSEMBLY_HINT"

/**
 * Structure used by clients/apps/etc to specify how they want their assembly
 * cooked. Thus, 'hint'.
 */
struct assembly_hint
{
    int num_gpus; /* XXX not used; vGPUs per process */
    enum hint_nic_type nic_type; /* XXX not used; network for remote vGPU */

    /* XXX What does this field assist with? I forgot already */
    unsigned int mpi_group; /* 0 = not mpi, else some number > 0 */

    enum hint_policy policy;
    size_t batch_size; /* 0 = max, else some power of two */

    // TODO

    // Attribute descriptors
    // Per GPU: ECC, mem size, mem bandwidth, mem clock, parallelization
    // Per assembly: throughput, latency, capacity

    // int num_cpus?
};

extern struct assembly_hint assembly_default_hint;

/**
 * Data type representing a key used to export and import an assembly across
 * processes.
 */
typedef uuid_t assembly_key_uuid;

/*-------------------------------------- FUNCTIONS ---------------------------*/

static inline
const char *node_type_str(enum node_type t)
{
    switch (t) {
        case NODE_TYPE_MAIN: return "NODE_TYPE_MAIN";
        case NODE_TYPE_MINION: return "NODE_TYPE_MINION";
        case NODE_TYPE_MAPPER: return "NODE_TYPE_MAPPER";
        default: return "NODE_TYPE Unknown or Invalid";
    }
}

/* because i hate string operations in C
 * why can't we just do string == string and be done with this nonsense */
static inline bool
str_eq(const char *a, const char *b, size_t len)
{
    return 0 == strncmp(a, b, len);
}

static inline const char *
assembly_hint_enum2nic(enum hint_nic_type t)
{
    switch(t)
    {
        case HINT_USE_ETH:
            return HINT_ETH_STR;
        case HINT_USE_IB:
            return HINT_IB_STR;
        default:
            return "unknown";
    }
}

static inline enum hint_policy
assembly_hint_policy2enum(const char *policy)
{
    if (!policy)
        return HINT_ENUM_POLICY_INVALID;

    if (strstr(policy, HINT_FIELD_POLICY_REMOTEONLY))
        return HINT_ENUM_POLICY_REMOTEONLY;

    else if (strstr(policy, HINT_FIELD_POLICY_LOCALFIRST))
        return HINT_ENUM_POLICY_LOCALFIRST;

    else
        return HINT_ENUM_POLICY_INVALID;
}

static inline const char *
assembly_hint_enum2policy(enum hint_policy p)
{
    switch(p)
    {
        case HINT_ENUM_POLICY_LOCALFIRST:
            return HINT_FIELD_POLICY_LOCALFIRST;
        case HINT_ENUM_POLICY_REMOTEONLY:
            return HINT_FIELD_POLICY_REMOTEONLY;
        default:
            return "unknown";
    }
}

int assembly_runtime_init(enum node_type type, const char *main_ip);
int assembly_runtime_shutdown(void);

asmid_t assembly_request(const struct assembly_hint *hint, pid_t pid);
int assembly_teardown(asmid_t id);

int assembly_num_vgpus(asmid_t id);

int assembly_read_hint(struct assembly_hint *hint);

/**
 * Implement an assembly on the cluster (construct the data paths). Must be
 * called on valid assembly IDs returned from assembly_request which have not
 * been torn down.
 *
 * @return  -EHOSTDOWN  A remote node hosting a vgpu is dead
 *          -ENETDOWN   Network tanked itself
 *          -EINVAL     id isn't associated with any assembly
 *          -EEXIST     assembly was already mapped
 *          zero        success
 */
int assembly_map(asmid_t id);

int assembly_vgpu_is_local(asmid_t id, int vgpu_id, bool *answer);

/**
 * Export an assembly returned from assembly_request for import by another
 * process. Assemblies can only be exported once.
 *
 * @param   id      ID of the assembly to export
 * @param   uuid    UUID identifying the export instance
 * @return  zero    success
 *          -EIO    Couldn't complete the export
 */
int assembly_export(asmid_t id, assembly_key_uuid uuid);

/**
 * Import an assembly that was exported by another process.
 *
 * @param   id      UUID identifying the exported instance
 * @param   uuid    Assembly returned by a previous call to export. Unique to a
 *                  specific assembly.
 * @return  -EINVAL Invalid key or bad pointer
 *          -EIO    Couldn't complete the import
 *          zero    success
 */
int assembly_import(asmid_t *id, const assembly_key_uuid uuid);

//! Print human-readable format of an assembly configuration.
void assembly_print(asmid_t id);

/* Associated CUDA interface functions, called directly from interposer or
 * application */

cudaError_t assm_cudaBindTexture(size_t*, const struct textureReference*, const void*, const struct cudaChannelFormatDesc*, size_t);
cudaError_t assm_cudaBindTextureToArray(const struct textureReference*, const struct cudaArray*,const struct cudaChannelFormatDesc*);
cudaError_t assm_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t, cudaStream_t);
cudaError_t assm_cudaDriverGetVersion(int*);
cudaError_t assm_cudaFreeArray(struct cudaArray*);
cudaError_t assm_cudaFreeHost(void*);
cudaError_t assm_cudaFree(void*);
cudaError_t assm_cudaFuncGetAttributes(struct cudaFuncAttributes*, const char*);
cudaError_t assm_cudaGetDeviceCount(int*);
cudaError_t assm_cudaGetDevice(int*);
cudaError_t assm_cudaGetDeviceProperties(struct cudaDeviceProp*, int);
cudaError_t assm_cudaGetLastError(void);
cudaError_t assm_cudaGetTextureReference(const struct textureReference**, const char*);
cudaError_t assm_cudaHostAlloc(void**, size_t, unsigned int);
cudaError_t assm_cudaLaunch(const char*);
cudaError_t assm_cudaMallocArray(struct cudaArray**, const struct cudaChannelFormatDesc*, size_t, size_t, unsigned int);
cudaError_t assm_cudaMallocPitch(void**, size_t*, size_t, size_t);
cudaError_t assm_cudaMalloc(void**, size_t);
cudaError_t assm_cudaMemcpyAsync(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
cudaError_t assm_cudaMemcpyFromSymbol(void*, const char*, size_t, size_t, enum cudaMemcpyKind);
cudaError_t assm_cudaMemcpyToArray(struct cudaArray*, size_t, size_t, const void*, size_t, enum cudaMemcpyKind);
cudaError_t assm_cudaMemcpyToSymbolAsync(const char*, const void*, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
cudaError_t assm_cudaMemcpyToSymbol(const char*, const void*, size_t, size_t, enum cudaMemcpyKind);
cudaError_t assm_cudaMemcpy(void*, const void*, size_t, enum cudaMemcpyKind);
cudaError_t assm_cudaMemGetInfo(size_t*, size_t*);
cudaError_t assm_cudaMemset(void*, int, size_t);
cudaError_t assm_cudaRuntimeGetVersion(int*);
cudaError_t assm_cudaSetDeviceFlags(unsigned int);
cudaError_t assm_cudaSetDevice(int);
cudaError_t assm_cudaSetupArgument(const void*, size_t, size_t);
cudaError_t assm_cudaSetValidDevices(int*, int);
cudaError_t assm_cudaStreamCreate(cudaStream_t*);
cudaError_t assm_cudaStreamDestroy(cudaStream_t);
cudaError_t assm_cudaStreamQuery(cudaStream_t);
cudaError_t assm_cudaStreamSynchronize(cudaStream_t);
cudaError_t assm_cudaThreadExit(void);
cudaError_t assm_cudaThreadSynchronize(void);

void** assm__cudaRegisterFatBinary(void*);
void   assm__cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*);
void   assm__cudaRegisterTexture(void**, const struct textureReference*, const void**, const char*, int, int, int);
void   assm__cudaRegisterVar(void**, char*, char*, const char*, int, int, int, int);
void   assm__cudaUnregisterFatBinary(void** fatCubinHandle);

const char* assm_cudaGetErrorString(cudaError_t);
struct cudaChannelFormatDesc assm_cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind);

#endif

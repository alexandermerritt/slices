/**
 * @file types.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-20
 * @brief Data types common across areas of the assembly state.
 */
#ifndef _TYPES_H
#define _TYPES_H

// System includes
#include <stdbool.h>

// CUDA includes
#include <cuda_runtime_api.h>

// Project includes
#include <cuda/ops.h>
#include <cuda/rpc.h>
#include <util/list.h>

/*-------------------------------------- TYPES -------------------------------*/

//! Length of IP addresses, hostnames, etc
#define HOST_LEN	64

//! Maximum number of threads that can setDevice to a vgpu
#define MAX_SET_TIDS 	4

enum vgpu_fixation
{
	VGPU_LOCAL = 1,
	VGPU_REMOTE
};

/**
 * State associating a vgpu with a pgpu.
 */
struct vgpu_mapping
{
	enum vgpu_fixation fixation; //! Type, used by assembly_rpc
	int vgpu_id; //! Virtual ID given to the application
	int pgpu_id; //! Physical gpu ID assigned by the local driver on hostname
	char hostname[HOST_LEN]; //! Location of physical GPU
	char ip[HOST_LEN]; //! Location of physical GPU
	struct cudaDeviceProp cudaDevProp; //! Properties of physical GPU

	//! threads which have called setDevice on this vgpu, probably won't see
	//! more than one thread set to the same GPU ever, could remove as array
	pthread_t set_tids[MAX_SET_TIDS];
	int num_set_tids;

	//! Function jump table. Execute locally or via RPC.
	struct cuda_ops ops;

	//! Connection state for remote data/control paths
	struct cuda_rpc *rpc;
};

//! Maximum number of vgpu mappings in an assembly.
#define MAX_VGPUS	16

/**
 * State that represents an assembly once created.
 *
 * TODO Should we store the fat binary and other vars/funcs/textures in this
 * structure? Currently, it is being stored in localsink.c. But that is
 * temporary.
 */
struct assembly
{
	struct list_head link;
	asmid_t id;
	int num_gpus;
	struct vgpu_mapping mappings[MAX_VGPUS];
	int driverVersion, runtimeVersion;
	struct fatcubins *cubins; // CUDA CUBIN state maintained on local node
	//! Whether this assembly was mapped. This value is only updated in the
	//! process which calls assembly_map (thus it will not be updated across
	//! address spaces).
	bool mapped;
	//! The original hint used to compose this assembly.
	struct assembly_hint hint;
};

#define PARTICIPANT_MAX_GPUS	4
#define PARTICIPANT_MAX_NICS	4

#define for_each_assembly(assm,list) \
	list_for_each_entry(assm,list,link)

/**
 * State describing a node that has registered to participate in the assembly
 * runtime. Minions send RPCs to the main node to register. Not using any
 * pointers in this structure as that eliminates the need to serialize it when
 * sending it as part of an RPC.
 */
struct node_participant
{
	struct list_head link;
	enum node_type type;

	// Network information
	char hostname[HOST_LEN];
	char ip[PARTICIPANT_MAX_NICS][HOST_LEN];
	char nic_name[PARTICIPANT_MAX_NICS][HOST_LEN];
	int num_nics;

	// GPU information
	int num_gpus;
	struct cudaDeviceProp dev_prop[PARTICIPANT_MAX_GPUS];
	int driverVersion, runtimeVersion;
};

#define for_each_node(node,list) \
	list_for_each_entry(node,list,link)

#endif	/* _TYPES_H */

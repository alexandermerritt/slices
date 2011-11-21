/**
 * @file types.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-20
 * @brief Data types common across areas of the assembly state.
 */
#ifndef _TYPES_H
#define _TYPES_H

// System includes

// CUDA includes
#include <cuda_runtime_api.h>

// Project includes
#include <util/list.h>

/*-------------------------------------- TYPES -------------------------------*/

//! Length of IP addresses, hostnames, etc
#define HOST_LEN	255

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
	enum vgpu_fixation fixation;

	int vgpu_id; //! vgpu ID given to the application
	char vgpu_hostname[HOST_LEN]; // FIXME this will be different for all MPI ranks
	char vgpu_ip[HOST_LEN]; // FIXME this will be different for all MPI ranks

	int pgpu_id; //! physical gpu ID assigned by the local driver
	char pgpu_hostname[HOST_LEN];
	char pgpu_ip[HOST_LEN];
	struct cudaDeviceProp cudaDevProp;

	//! threads which have called setDevice on this vgpu, probably won't see
	//! more than one thread set to the same GPU ever, could remove as array
	pthread_t set_tids[MAX_SET_TIDS];
	int num_set_tids;
};

/**
 * Maximum number of vgpus in an assembly. We have this as the assembly
 * structure is passed from the MAIN node to MINION nodes. It must be a flat
 * structure to avoid marshalling it.
 */
#define MAX_VGPUS	24

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
	// TODO array of static gpu properties
	// TODO link to dynamic load data?
	// TODO location of node containing application
};

#define PARTICIPANT_MAX_GPUS	4
#define PARTICIPANT_MAX_NICS	4

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

	// GPU information
	int num_gpus;
	struct cudaDeviceProp dev_prop[PARTICIPANT_MAX_GPUS];
	int driverVersion, runtimeVersion;
};

#endif	/* _TYPES_H */

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

// Project includes
#include <packetheader.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

#define ASSEMBLY_SHMGRP_KEY "cudarpc"

/**
 * Distinguish between the centralized location where state of all assemblies is
 * located, or an instance on another node (minion) that forwards RPCs to the
 * main node to carry out the work. The third option is to initialize the
 * library to be a mapper, meaning it will import exported assemblies created by
 * a MAIN or MINION node for execution.
 */
enum node_type
{
	NODE_TYPE_MAIN = 0,
	NODE_TYPE_MINION,
	NODE_TYPE_MAPPER,
	NODE_TYPE_INVALID	//! Used to compare against valid type values
};

#define INVALID_ASSEMBLY_ID		(0UL)
#define VALID_ASSEMBLY_ID(id)	((id) != INVALID_ASSEMBLY_ID)

typedef unsigned long asmid_t;

/**
 * Structure used by clients/apps/etc to specify how they want their assembly
 * cooked. Thus, 'hint'.
 */
struct assembly_hint
{
	int num_gpus;

	// TODO

	// Attribute descriptors
	// Per GPU: ECC, mem size, mem bandwidth, mem clock, parallelization
	// Per assembly: throughput, latency, capacity

	// int num_cpus?
};

/**
 * Data type representing a key used to export and import an assembly across
 * processes.
 */
typedef uuid_t assembly_key_uuid;

/*-------------------------------------- FUNCTIONS ---------------------------*/

int assembly_runtime_init(enum node_type type, const char *main_ip);
int assembly_runtime_shutdown(void);

asmid_t assembly_request(const struct assembly_hint *hint);
int assembly_teardown(asmid_t id);

int assembly_num_vgpus(asmid_t id);
int assembly_vgpu_is_remote(asmid_t id, int vgpu);
int assembly_set_batch_size(asmid_t id, int vgpu, unsigned int size);

/**
 * Implement an assembly on the cluster (construct the data paths). Must be
 * called on valid assembly IDs returned from assembly_request which have not
 * been torn down.
 *
 * @return 	-EHOSTDOWN	A remote node hosting a vgpu is dead
 * 			-ENETDOWN	Network tanked itself
 * 			-EINVAL		id isn't associated with any assembly
 * 			-EEXIST		assembly was already mapped
 * 			zero		success
 */
int assembly_map(asmid_t id);
int assembly_rpc(asmid_t id, int vgpu_id, struct cuda_packet *pkt);

/**
 * Export an assembly returned from assembly_request for import by another
 * process. Assemblies can only be exported once.
 *
 * @param	id		ID of the assembly to export
 * @param	uuid	UUID identifying the export instance
 * @return	zero	success
 * 			-EIO	Couldn't complete the export
 */
int assembly_export(asmid_t id, assembly_key_uuid uuid);

/**
 * Import an assembly that was exported by another process.
 *
 * @param	id		UUID identifying the exported instance
 * @param	uuid	Assembly returned by a previous call to export. Unique to a
 * 					specific assembly.
 * @return	-EINVAL	Invalid key or bad pointer
 * 			-EIO	Couldn't complete the import
 * 			zero	success
 */
int assembly_import(asmid_t *id, const assembly_key_uuid uuid);

//! Print human-readable format of an assembly configuration.
void assembly_print(asmid_t id);

#endif

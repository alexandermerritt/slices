/**
 * @file assembly.h
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
 */
#ifndef _ASSEMBLY_H
#define _ASSEMBLY_H

#include <packetheader.h>

/**
 * Distinguish between the centralized location where state of all assemblies is
 * located, or an instance on another node (minion) that forwards RPCs to the
 * main node to carry out the work.
 */
enum node_type
{
	NODE_TYPE_MAIN = 0,
	NODE_TYPE_MINION,
	NODE_TYPE_INVALID	//! Used to compare against valid type values
};

#define INVALID_ASSEMBLY_ID		(0UL)
#define VALID_ASSEMBLY_ID(id)	((id) != INVALID_ASSEMBLY_ID)

typedef unsigned long asmid_t;

/**
 * Structure used by clients/apps/etc to specify how they want their assembly
 * cooked. Thus, 'hint'.
 */
struct assembly_cap_hint
{
	int num_gpus;

	// TODO

	// Attribute descriptors
	// Per GPU: ECC, mem size, mem bandwidth, mem clock, parallelization
	// Per assembly: throughput, latency, capacity

	// int num_cpus?
};

int assembly_runtime_init(enum node_type type);
int assembly_runtime_shutdown(void);

asmid_t assembly_request(const struct assembly_cap_hint *hint);
int assembly_num_vgpus(asmid_t id);
int assembly_vgpu_is_remote(asmid_t id, int vgpu);
int assembly_set_batch_size(asmid_t id, int vgpu, unsigned int size);
int assembly_rpc(asmid_t id, int vgpu_id, volatile struct cuda_packet *pkt);
int assembly_teardown(asmid_t id);

#endif

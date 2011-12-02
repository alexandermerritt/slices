/**
 * @file assemble.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-30
 * @brief This file contains the infrastructure and algorithms for composing
 * assemblies given available nodes, existing assemblies, dynamic monitoring of
 * node utilization and a hint structure provided by the user.
 *
 * TODO Define different composers based on certain flags in the hint
 * structure---policies or whatnot---that require a different search algorithm.
 * These composers will require access to the participant and assembly lists, as
 * well as the definitions of a hint, vgpu_mapping, assemblies, etc.
 */

// System includes
#include <string.h>

// Other project includes
// TODO Incorporate monitoring API

// Project includes
#include <assembly.h>
#include <debug.h>
#include <util/list.h>

// Directory-immediate includes
#include "types.h"

/*-------------------------------------- STRUCTURES --------------------------*/

/**
 * Global state we're working with. When a request for an assembly is made, that
 * request must come from somewhere; that is represented by 'hostname' (knowing
 * this allows us to determine if vgpus are local or remote). The contents of
 * this structure will be different each time a request is made with the
 * addition of new assemblies we create, or nodes that come and go, or with
 * different hosts requesting assembles. It represents the known state to exist
 * at the time of the request.
 */
struct global
{
	const struct assembly_hint *hint; //! Application-provided hints
	const char *hostname; //! Host requesting an assembly
	const struct list_head *nlist; //! Existing node_participants
	const struct list_head *alist; //! Existing assemblies
};

/**
 * Simple type to identify a GPU in a cluster.
 */
struct gpu
{
	int id; //! CUDA device ID
	const struct node_participant *node; //! Points within global.nlist
};

/*-------------------------------------- FUNCTIONS ---------------------------*/

// TODO
// Find nodes w/ specific GPU, GPU properties
// Find non-mapped assembly

static inline bool
vgpus_same_host(const struct vgpu_mapping *v1, const struct vgpu_mapping *v2)
{
	return (strncmp(v1->hostname, v2->hostname, HOST_LEN) == 0);
}

static inline bool
node_is_remote(const struct global *g, const struct node_participant *n)
{
	return (strncmp(g->hostname, n->hostname, HOST_LEN) != 0);
}

/**
 * Locate the first remote node we find in the cluster, relative
 * to whom is requesting the assembly. We assume the node list is not empty, and
 * that each node in the list has at least one gpu.
 *
 * @return	true	A gpu was found
 * 			false	No gpus are remote with respect to the requester
 */
static bool
find_first_remote_gpu(const struct global *global, struct gpu *gpu)
{
	bool node_found = false;
	const struct node_participant *node = NULL;
	for_each_node(node, global->nlist) {
		if (node_is_remote(global, node)) {
			node_found = true;
			break;
		}
	}
	if (!node_found)
		return false;
	gpu->id = 0; // first gpu in 'node'
	gpu->node = node;
	return true;
}

/**
 * Get the first gpu on the local node. The 'local' node is always assumed to
 * exist and to contain 1+ gpus.
 */
static void
find_first_local_gpu(const struct global *global, struct gpu *gpu)
{
	bool node_found = false;
	const struct node_participant *node = NULL;
	for_each_node(node, global->nlist) {
		if (!node_is_remote(global, node)) {
			node_found = true;
			break;
		}
	}
	BUG(!node_found);
	gpu->id = 0;
	gpu->node = node;
}

static inline int
fix_assm_size(int size)
{
	int new_size = size;
	if (size == 0) {
		new_size = 1;
	} else if (size > MAX_VGPUS) {
		new_size = MAX_VGPUS;
	}
	return new_size;
}

// caller is responsible for setting vgpu_id, as it depends on the assembly
// composition
static void
set_vgpu_mapping(const struct global *global,
		const struct gpu *gpu, struct vgpu_mapping *vgpu)
{
	const struct node_participant *node = gpu->node;
	vgpu->fixation =
		(node_is_remote(global, node) ? VGPU_REMOTE : VGPU_LOCAL);
	vgpu->pgpu_id = gpu->id;
	strncpy(vgpu->hostname, node->hostname, HOST_LEN);
	strncpy(vgpu->ip, node->ip[1], HOST_LEN); // FIXME Figure out NIC
	memcpy(&vgpu->cudaDevProp, &node->dev_prop[gpu->id],
			sizeof(struct cudaDeviceProp));
}

/*-------------------------------------- ENTRY -------------------------------*/

//! Entry point to this file, invoked from assembly.c::__compose_assembly.
struct assembly *
__do_compose_assembly(
		const struct assembly_hint *hint,
		const char *hostname,
		const struct list_head *node_list,
		const struct list_head *assembly_list)
{
	struct global global;
	struct assembly *assm = NULL;
	struct gpu gpu;

	assm = calloc(1, sizeof(*assm));
	if (!assm)
		goto fail;

	/* We must be careful iterating over nlist and alist using the macros in
	 * list.h. The heads of these lists themselves do not reside within an
	 * instance of the object the list is intended for, but are contained within
	 * some other global structure of another type. Thus when we iterate over
	 * the list, we must make sure that if the ENTIRE list is iterated (e.g.
	 * we're searching for a particular element and end up traversing it
	 * entirely), we do NOT dereference the iterator pointer, as it will be cast
	 * to an object within the structure containing the head of the list. We
	 * must instead maintain a boolean indicating if an element was found or
	 * not.
	 */
	global.hint = hint;
	global.hostname = hostname;
	global.nlist = node_list;
	global.alist = assembly_list;

	// Determine assembly size
	assm->num_gpus = 1;

	// Locate GPUs to use. Look for a remote node, but return the local node if
	// none was found.
	if (!find_first_remote_gpu(&global, &gpu)) {
		find_first_local_gpu(&global, &gpu);
	}

	// NOTE: If no GPU can be found remotely, this function should always
	// default to returning local GPUs in their place.

	BUG(!gpu.node);

	// Install mappings
	set_vgpu_mapping(&global, &gpu, &assm->mappings[0]);
	assm->mappings[0].vgpu_id = 0;

	return assm;

fail:
	if (assm) free(assm);
	return NULL;
}

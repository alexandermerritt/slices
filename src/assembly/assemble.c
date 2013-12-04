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

#define _GNU_SOURCE /* for strchrnul */

// System includes
#include <stdbool.h>
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
	struct assembly_hint *hint; //! Application-provided hints
	char *hostname; //! Host requesting an assembly
	struct list_head *nlist; //! Existing node_participants
	struct list_head *alist; //! Existing assemblies
};

/**
 * Simple type to identify a GPU in a cluster.
 */
struct gpu
{
	int id; //! CUDA device ID
	struct node_participant *node; //! Points within global.nlist
};

/*-------------------------------------- FUNCTIONS ---------------------------*/

// TODO
// Find nodes w/ specific GPU, GPU properties
// Find non-mapped assembly

static inline bool
vgpus_same_host(struct vgpu_mapping *v1, struct vgpu_mapping *v2)
{
	return (strncmp(v1->hostname, v2->hostname, HOST_LEN) == 0);
}

static inline bool
node_is_remote(struct global *g, struct node_participant *n)
{
	return (strncmp(g->hostname, n->hostname, HOST_LEN) != 0);
}

/* Search through all assemblies for the provided mpi_group to see if any remote
 * vgpus within them map to the provided node. This search is to assist in
 * avoiding mapping >1 remote vGPUs with the same mpi_group to the same node, as
 * the remote sink is implemented as a single process.
 */
static bool
remote_vgpu_mpi_group_conflict(struct global *global,
        unsigned int mpi_group,
        struct node_participant *node)
{
    struct assembly *assm;
    struct vgpu_mapping *vgpu;
    int id;
    list_for_each_entry(assm, global->alist, link) {
        if (assm->hint.mpi_group != mpi_group)
            continue;
        for (id = 0; id < assm->num_gpus; id++) {
            vgpu = &assm->mappings[id];
            if (!str_eq(vgpu->hostname, node->hostname, HOST_LEN))
                continue; /* vgpu doesn't map to us */
            if (vgpu->fixation == VGPU_LOCAL)
                continue; /* only care about incoming vgpu mappings */
            /* we found an assembly of the same MPI group which has a remote
             * vGPU mapped to this node */
            return true;
        }
    }
    /* an exhaustive search of all assemblies of the provided MPI group shows
     * none have remote vGPUs which map to this node */
    return false;
}

/**
 * "First" is relative of course, it is the first in the list, most likely the
 * first node that joined.
 */
static bool
find_first_remote_node(
		struct global *global,
		struct node_participant **remote_node)
{
	bool node_found = false;
	struct node_participant *node = NULL;
	for_each_node(node, global->nlist) {
		if (node_is_remote(global, node)) {
			node_found = true;
			break;
		}
	}
	if (!node_found)
		return false;
	*remote_node = node;
	return true;
}

/* HACK skip nodes not explicitly listed in the override env variable */
/* we assume nodes in the env var are short DNS names with no . */
static bool
in_nodelist(const char *hostname)
{
    char dns[HOST_NAME_MAX];
    const char *env_nodelist = getenv("REMOTE_NODE_LIST");

    printd(DBG_DEBUG, "REMOTE_NODE_LIST defined to '%s'\n",
            env_nodelist);

    // if not defined, all nodes valid
    if (!env_nodelist)
        return true;
    memset(dns, 0, HOST_NAME_MAX * sizeof(*dns));

    char *save, *entry, *str = strdup(env_nodelist);
    if (!str) { fprintf(stderr, "oom\n"); abort(); }

    strncpy(dns, hostname, HOST_NAME_MAX-1);
    *strchrnul(dns, '.') = '\0';
    printd(DBG_DEBUG, "comparing against '%s'\n", dns);

    bool found = false;
    entry = strtok_r(str, ":", &save);
    do {
        if (strncmp(dns, entry, strlen(dns)) == 0)
            found = true;
    } while (!found && (entry = strtok_r(NULL, ":", &save)));

    free(str);

    return found;
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
find_first_remote_gpu(struct global *global, struct gpu *gpu,
        unsigned int mpi_group)
{
	struct node_participant *node = NULL;

	for_each_node(node, global->nlist) {
		if (!node_is_remote(global, node))
            continue;

        /* if this is a multi-process program .. */
        if (mpi_group != 0) {
            printd(DBG_DEBUG, "this is a multi-process program\n");
            /* .. additionally skip nodes which have remote vgpus mapped to it
             * by other processes of the same program */
            if (remote_vgpu_mpi_group_conflict(global, mpi_group, node)) {
                printd(DBG_DEBUG, "grp %u has remote vgpu which"
                       " maps to %s, skipping node\n",
                        mpi_group, node->hostname);
                continue;
            }
        }

        if (!in_nodelist(node->hostname))
            continue;

        goto found;
    }

    printd(DBG_INFO, "No node found\n");
    return false; /* nerrrrp! */

found:
    printd(DBG_INFO, "Found 0@%s\n", node->hostname);
    gpu->id = 0;
    gpu->node = node;
	return true;
}

/**
 * Looks for N gpus on a single remote node. If the first remote node found does
 * not have at least N gpus, request fails. Caller should decrease amount
 * required and request again. Also fails if no remote node exists.
 *
 * N is found in the hint within the global state. 'gpus' must be an array of
 * struct gpu able to store at least hint.num_gpus.
 */
static bool
find_first_N_remote_gpus(struct global *global, struct gpu *gpus)
{
	int N = global->hint->num_gpus;
	struct node_participant *node;
	if (!find_first_remote_node(global, &node))
		return false;
	if (node->num_gpus < N)
		return false;
	int vgpu_id = 0, gpu;
	for (gpu = 0; gpu < N; gpu++) {
		gpus[gpu].id = vgpu_id++;
		gpus[gpu].node = node;
	}
	return true;
}

static void
find_local_node(
		struct global *global,
		struct node_participant **local_node)
{
	bool node_found = false;
	struct node_participant *node = NULL;
	for_each_node(node, global->nlist) {
		if (!node_is_remote(global, node)) {
			node_found = true;
			break;
		}
	}
	BUG(!node_found);
	*local_node = node;
}

/**
 * Get the first gpu on the local node. The 'local' node is always assumed to
 * exist and to contain 1+ gpus.
 */
static void
find_first_local_gpu(struct global *global, struct gpu *gpu)
{
	struct node_participant *node = NULL;
	find_local_node(global, &node);
	gpu->id = 0;
	gpu->node = node;
}

static inline bool
find_unmapped_local_gpu(struct global *global, struct gpu *gpu)
{
	struct node_participant *node;
    int id;

	find_local_node(global, &node);

    for (id = 0; id < node->num_gpus; id++)
        if (node->gpu_mapped[id] == 0)
            break;

    if (id >= node->num_gpus)
        return false; /* nerp! */

    gpu->id = id;
    gpu->node = node;

    return true; /* yerp! */
}

/* mpi_group used to avoid creating >1 remote vGPUs to the same node. 0 means
 * this restriction is ignored (non-MPI application). */
static inline bool
find_unmapped_remote_gpu(struct global *global, struct gpu *gpu,
        unsigned int mpi_group)
{
    struct node_participant *node;
    int id;

    for_each_node(node, global->nlist) {
        if (!node_is_remote(global, node))
            continue;

        /* if this is a multi-process program .. */
        if (mpi_group != 0) {
            printd(DBG_DEBUG, "this is a multi-process program\n");
            /* .. additionally skip nodes which have remote vgpus mapped to it
             * by other processes of the same program */
            if (remote_vgpu_mpi_group_conflict(global, mpi_group, node)) {
                printd(DBG_DEBUG, "grp %u has remote vgpu which"
                        " maps to %s, skipping node\n",
                        mpi_group, node->hostname);
                continue;
            }
        }

        if (!in_nodelist(node->hostname))
            continue;

        /* locate a gpu on this node which has nothing mapped to it */
        for (id = 0; id < node->num_gpus; id++)
            if (node->gpu_mapped[id] == 0)
                goto found;
        printd(DBG_DEBUG, "node %s has no available GPUs\n", node->hostname);
    }

    printd(DBG_INFO, "No node found\n");
    return false; /* nerrrrp! */

found:
    printd(DBG_INFO, "Found %d@%s\n", id, node->hostname);
    gpu->id = id;
    gpu->node = node;
    return true;
}

/** May return false if request is too large. */
static bool
find_first_N_local_gpus(struct global *global, struct gpu *gpus)
{
	int N = global->hint->num_gpus;
	struct node_participant *node;
	find_local_node(global, &node);
    if (N == 0) {
        printd(DBG_ERROR, "requesting 0 gpus?\n");
        return false;
    }
	if (node->num_gpus < N) {
        printd(DBG_WARNING, "node %s doesn't have enough GPUS (%d)"
                " for request of %d\n",
                node->hostname, node->num_gpus, N);
		return false;
    }
	int vgpu_id = 0, gpu;
	for (gpu = 0; gpu < N; gpu++) {
		gpus[gpu].id = vgpu_id++;
		gpus[gpu].node = node;
	}
	return true;
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
set_vgpu_mapping(struct global *global,
		struct gpu *gpu, struct vgpu_mapping *vgpu)
{
	struct node_participant *node = gpu->node;
	int nic;
	vgpu->fixation =
		(node_is_remote(global, node) ? VGPU_REMOTE : VGPU_LOCAL);
	vgpu->pgpu_id = gpu->id;
    node->gpu_mapped[gpu->id]++;
    printd(DBG_DEBUG, "%d@%s now has %d mappings\n", gpu->id, node->hostname,
            node->gpu_mapped[gpu->id]);
	strncpy(vgpu->hostname, node->hostname, HOST_LEN);
	// find the NIC specified in the hint
	// TODO find a better way to store this information
	nic = 0;
	enum hint_nic_type type = global->hint->nic_type;
	char *nic_str_cmp;
	if (type == HINT_USE_ETH)
		nic_str_cmp = HINT_ETH_STR;
	else if (type == HINT_USE_IB)
		nic_str_cmp = HINT_IB_STR;
	else
		BUG(1);
	while (nic < PARTICIPANT_MAX_NICS || nic < (node->num_nics)) {
		if (strncmp(node->nic_name[nic], nic_str_cmp, strlen(nic_str_cmp)) == 0) {
			strncpy(vgpu->ip, node->ip[nic], HOST_LEN);
			printd(DBG_DEBUG, "using pGPU %d on %s\n",
                    vgpu->pgpu_id, node->hostname);
			break;
		}
		nic++;
	}
	BUG(nic >= node->num_nics);
	memcpy(&vgpu->cudaDevProp, &node->dev_prop[gpu->id],
			sizeof(struct cudaDeviceProp));
}

/*-------------------------------------- ENTRY -------------------------------*/

//! Entry point to this file, invoked from assembly.c::__compose_assembly.
struct assembly *
__do_compose_assembly(
		struct assembly_hint *hint,
		char *hostname,
		struct list_head *node_list,
		struct list_head *assembly_list)
{
	struct global global;
	struct assembly *assm = NULL;
	struct gpu *gpus = NULL;
	int gpus_granted = 0;

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
	gpus_granted = hint->num_gpus;

	gpus = calloc(hint->num_gpus, sizeof(*gpus));
	if (!gpus) goto fail;

    switch (hint->policy) {

        case HINT_ENUM_POLICY_LOCALFIRST:
        {
            gpus_granted = 1;
            if (find_unmapped_local_gpu(&global, &gpus[0]))
                break;
            printd(DBG_INFO, "No unmapped local GPUs available, trying remote\n");
            if (find_unmapped_remote_gpu(&global, &gpus[0], hint->mpi_group))
                break;
            printd(DBG_INFO, "No unmapped remote GPUs available, assigning local\n");
            find_first_local_gpu(&global, &gpus[0]); /* always succeeds */
        }
        break;

        case HINT_ENUM_POLICY_REMOTEONLY:
        {
            gpus_granted = 1;
            if (find_unmapped_remote_gpu(&global, &gpus[0], hint->mpi_group))
                break;
            printd(DBG_INFO, "No _unmapped_ remote GPUs available,"
                    " trying any remote GPU\n");
            if (find_first_remote_gpu(&global, &gpus[0], hint->mpi_group))
                break;
            /* TODO hm.. nothing available remotely.. avoid catastrophic failure
            * by assigning a local gpu, but warn */
            printd(DBG_INFO, "No _remote_ GPUs available, assigning local\n");
            find_first_local_gpu(&global, &gpus[0]);
        }
        break;

        default:
            BUG(1);
    }

	// Determine assembly size
	assm->num_gpus = gpus_granted;

	// Should always have at least one GPU, its vgpu_id being zero.
	BUG(!gpus[0].node);

	// Install mappings
	int gpu_id;
	for (gpu_id = 0; gpu_id < gpus_granted; gpu_id++) {
		set_vgpu_mapping(&global, &gpus[gpu_id], &assm->mappings[gpu_id]);
		assm->mappings[gpu_id].vgpu_id = gpu_id;
	}

	if (gpus) free(gpus);
	return assm;

fail:
	if (assm) free(assm);
	if (gpus) free(gpus);
	return NULL;
}

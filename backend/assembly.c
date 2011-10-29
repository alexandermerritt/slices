/**
 * @file assembly.c
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
 */

#include <string.h>
#include <cuda.h>

#include <assembly.h>
#include <debug.h>
#include <method_id.h>
#include <util/compiler.h>
#include <util/list.h>

// FIXME
// Defining prototypes here. Compiler complains despite including cuda.h
extern cudaError_t cudaGetDeviceCount(unsigned int* dev);
extern cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int dev);

/*-------------------------------------- INTERNAL STRUCTURES -----------------*/

enum vgpu_fixation
{
	VGPU_LOCAL = 0,
	VGPU_REMOTE
};

/**
 * State associating a vgpu with a pgpu.
 */
struct vgpu_mapping
{
	struct assembly *assm; //! Assembly holding this vgpu
	enum vgpu_fixation fixation;

	int vgpu_id; //! vgpu ID given to the application
	char vgpu_hostname[255];
	char vgpu_ip[255];

	int pgpu_id; //! physical gpu ID assigned by the local driver
	char pgpu_hostname[255];
	char pgpu_ip[255];
	struct cudaDeviceProp cudaDevProp;
};

#define ASSEMBLY_MAX_MAPPINGS	24U

/**
 * State that represents an assembly once created.
 */
struct assembly
{
	struct list_head list;
	asmid_t id;
	unsigned int num_gpus;
	struct vgpu_mapping mappings[ASSEMBLY_MAX_MAPPINGS];
	// TODO array of static gpu properties
	// TODO link to dynamic load data?
	// TODO location of node containing application
};

#define PARTICIPANT_MAX_GPUS	4

/**
 * State describing a node that has registered to participate in the assembly
 * runtime. Minions send RPCs to the main node to register.
 */
struct node_participant
{
	struct list_head list;
	char hostname[255];
	char ip[255];
	unsigned int num_gpus;
	struct cudaDeviceProp dev_prop[PARTICIPANT_MAX_GPUS];
	enum node_type type;
};

struct main_state
{
	//! Used by main node to assign global assembly IDs
	asmid_t next_asmid;

	//! List of node_participant structures represent available nodes.
	struct list_head participants;
};

struct minion_state
{
	// TODO
	// Location of main server.
};

/**
 * Assembly module internal state. Describes node configuration and contains set
 * of assemblies created.
 */
struct internals_state
{
	enum node_type type;
	pthread_mutex_t lock; //! Lock for changes to this struct
	struct list_head assembly_list;
	union node_specific {
		struct main_state main;
		struct minion_state minion;
	} n;
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct internals_state *internals = NULL;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

#define NEXT_ASMID	(internals->n.main.next_asmid++)

/**
 * Locate the assembly structure given the ID. Assumes locks governing the list
 * have been acquired by the caller.
 */
static struct assembly* find_assembly(asmid_t id)
{
	struct assembly *assm = NULL;
	list_for_each_entry(assm, &internals->assembly_list, list)
		if (assm->id == id)
			break;
	if (unlikely(!assm || assm->id != id))
		printd(DBG_ERROR, "unknown assembly id %lu\n", id);
	return assm;
}

/**
 * MAIN
 * Calls into the API will directly invoke the respective internal functions to
 * carry out the work. Spawn thread that listens and accepts incoming assembly
 * RPCs, which represent the API (public functions below) directly. The thread
 * directly calls the API functions as if it were some external entity
 * (essentially it is, acting as a proxy). As part of the registration by minion
 * nodes, their participation information (num gpus, etc) is added.
 *
 * MINION
 * Calls into the API get sent as RPCs to the main node. No proxy thread is
 * created, as this type of node does not accept incoming requests.
 *
 * The functions that go remote may include only request and teardown. Caching
 * assemblies per node will allow other calls to return more quickly. To
 * maintain consistency, it may asynchronously send a packet to the main node.
 */

/**
 * Figure out the composition of a new assembly given the hints, monitoring
 * state and currently existing assemblies. Assumes locks are held by caller.
 * This function will become quite large, I assume. If not, then this file will.
 */
static struct assembly* compose_assembly(const struct assembly_cap_hint *hint)
{
	struct node_participant *node = NULL;
	struct assembly *assm = calloc(1, sizeof(struct assembly));
	unsigned int vgpu_id;
	struct vgpu_mapping *vgpu;
	if (!assm) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&assm->list);
	assm->id = NEXT_ASMID;

	// TODO Simple 'compose' technique for starters: compare the list of
	// nodes/GPUs registered with me, locate ones not already mapped and create
	// the assembly based on that. Pretty much compose an assembly based on
	// exactly what the hint says. Later we can consider load, etc.

	// fix size of assembly
	if (hint->num_gpus == 0) {
		printd(DBG_WARNING, "Hint asks for zero GPUs; providing 1\n");
		assm->num_gpus = 1;
	} else if (hint->num_gpus > ASSEMBLY_MAX_MAPPINGS) {
		printd(DBG_WARNING, "Hint asks for %u GPUs; providing %u\n",
				hint->num_gpus, ASSEMBLY_MAX_MAPPINGS);
		assm->num_gpus = ASSEMBLY_MAX_MAPPINGS;
	} else {
		assm->num_gpus = hint->num_gpus;
	}

	// find physical GPUs and set each mapping
	list_for_each_entry(node, &internals->n.main.participants, list) {
		if (node->type != NODE_TYPE_MAIN) // FIXME don't limit to main node
			continue;
		for (vgpu_id = 0U; vgpu_id < node->num_gpus; vgpu_id++) {
			vgpu = &assm->mappings[vgpu_id];
			vgpu->assm = assm;
			vgpu->fixation = VGPU_LOCAL;
			vgpu->vgpu_id = vgpu_id;
			vgpu->pgpu_id = vgpu_id;
			memcpy(vgpu->vgpu_hostname, node->hostname, 255);
			memcpy(vgpu->pgpu_hostname, node->hostname, 255);
			memcpy(vgpu->vgpu_ip, node->ip, 255);
			memcpy(vgpu->pgpu_ip, node->ip, 255);
			memcpy(&vgpu->cudaDevProp, &node->dev_prop[vgpu_id],
					sizeof(struct cudaDeviceProp));
		}
	}
	if (vgpu_id <= assm->num_gpus) {
		printf("Warning: Could not assign all vGPUs\n");
	}

	// TODO For remote GPUs, construct data paths necessary to reach them.
	// After that, calls to assembly_rpc() should work.

	return assm;

fail:
	if (assm)
		free(assm);
	return NULL;
}

static int node_main_init(void)
{
	struct node_participant *p;
	int dev;
	cudaError_t cerr;

	internals->n.main.next_asmid = 1UL;
	INIT_LIST_HEAD(&internals->n.main.participants);

	p = calloc(1, sizeof(struct node_participant));
	if (!p) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&p->list);
	
	// FIXME don't hardcode, figure out at runtime
	strcpy(p->ip, "10.0.0.1");
	cerr = cudaGetDeviceCount(&p->num_gpus);
	if (cerr != cudaSuccess) {
		printd(DBG_ERROR, "error calling cudaGetDeviceCount: %d\n", cerr);
		goto fail;
	}
	printd(DBG_DEBUG, "node has %d GPUs\n", p->num_gpus);
	for (dev = 0; (unsigned int)dev < p->num_gpus; dev++) {
		cerr = cudaGetDeviceProperties(&p->dev_prop[dev], dev);
		if (cerr != cudaSuccess) {
			printd(DBG_ERROR, "error calling cudaGetDeviceProperties: %d\n",
					cerr);
			goto fail;
		}
	}
	strcpy(p->hostname, "ifrit"); // just for identification, not routing
	p->type = NODE_TYPE_MAIN;

	// no need to lock list, as nobody else exists at this point
	list_add(&p->list, &internals->n.main.participants);

	// TODO Spawn RPC thread.

	return 0;
fail:
	return -1;
}

static int node_main_shutdown(void)
{
	int err;
	struct node_participant *node_pos = NULL, *node_tmp = NULL;
	struct assembly *asm_pos = NULL, *asm_tmp = NULL;

	// Free participant list. We assume all minions have unregistered at this
	// point (meaning only one entry in the list).
	list_for_each_entry_safe(node_pos, node_tmp,
			&internals->n.main.participants, list) {
		if (node_pos->type == NODE_TYPE_MINION) { // oops...
			printd(DBG_ERROR, "minion still connected: TODO\n");
		}
		list_del(&node_pos->list);
		free(node_pos);
	}

	// Free assembly list. Again, we assume minions have shutdown their
	// assemblies, leaving this list empty.
	if (!list_empty(&internals->assembly_list)) {
		printd(DBG_ERROR, "Assemblies still exist!\n");
		fprintf(stderr, "Assemblies still exist!\n");
		list_for_each_entry_safe(asm_pos, asm_tmp,
				&internals->assembly_list, list) {
			err = assembly_teardown(asm_pos->id);
			if (err < 0)
				printd(DBG_ERROR, "could not teardown assembly %lu\n",
						asm_pos->id);
			list_del(&asm_pos->list);
			free(asm_pos);
		}
	}

	free(internals);
	internals = NULL;

	return 0;
}

static int node_minion_init(void)
{
	// TODO Register with main if we are a minion.
	fprintf(stderr, "Minion node not yet supported\n");
	return -1;
}

static int node_minion_shutdown(void)
{
	// TODO Register with main if we are a minion.
	fprintf(stderr, "Minion node not yet supported\n");
	return -1;
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int assembly_runtime_init(enum node_type type)
{
	int err;
	if (type >= NODE_TYPE_INVALID) {
		printd(DBG_ERROR, "invalid type: %d\n", type);
		goto fail;
	}
	if (internals) {
		printd(DBG_ERROR, "init already called\n");
		goto fail;
	}
	internals = calloc(1, sizeof(struct internals_state));
	if (!internals) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	internals->type = type;
	INIT_LIST_HEAD(&internals->assembly_list);

	// TODO Discover configuration of local node to add to participant data.

	if (type == NODE_TYPE_MINION) {
		err = node_minion_init();
		if (err < 0)
			goto fail;
	} else if (type == NODE_TYPE_MAIN) {
		err = node_main_init();
		if (err < 0)
			goto fail;
	}

	printd(DBG_INFO, "Assembly node configured as %s.\n",
			(type == NODE_TYPE_MAIN ? "main" : "minion"));

	return 0;

fail:
	return -1;
}

int assembly_runtime_shutdown(void)
{
	int err;
	if (!internals) {
		printd(DBG_WARNING, "assembly runtime not initialized\n");
		return 0;
	}
	if (internals->type == NODE_TYPE_MAIN)
		err = node_main_shutdown();
	else if (internals->type == NODE_TYPE_MINION)
		err = node_minion_shutdown();
	if (err < 0)
		goto fail;
	if (internals)
		free(internals);
	internals = NULL;
	return 0;

fail:
	printd(DBG_ERROR, "Could not shutdown\n");
	return -1;
}

int assembly_num_vgpus(asmid_t id)
{
	struct assembly *assm = NULL;
	unsigned int num_gpus = 0U;
	int err;
	if (id == 0U) {
		printd(DBG_ERROR, "Invalid assembly ID: 0\n");
		goto fail;
	}
	if (!internals) {
		printd(DBG_ERROR, "assembly runtime not initialized\n");
		goto fail;
	}
	// TODO If we are a minion node, we may need to send an RPC. If we cache
	// assemblies, we could check that first, but given the data sent (asmid and
	// int for size) is small, all that work may not be necessary.
	err = pthread_mutex_lock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not lock internals\n");
	list_for_each_entry(assm, &internals->assembly_list, list) {
		if (assm->id == id) {
			num_gpus = assm->num_gpus;
			break;
		}
	}
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not unlock internals\n");

	return num_gpus;

fail:
	return -1;
}

int assembly_vgpu_is_remote(asmid_t id, int vgpu)
{
	// check id, return state of cached assembly, maybe also send rpc
	return -1;
}

int assembly_set_batch_size(asmid_t id, int vgpu, unsigned int size)
{
	// check id, return state of cached assembly, maybe also send rpc
	// lock something
	return -1;
}

int assembly_rpc_nolock(asmid_t id, int vgpu, struct cuda_packet *pkt)
{
	// TODO May only be used by a sink directly. Idea is to not use any locking
	// when traversing assembly list, etc. This can only be done if sinks are
	// separate processes, as the assembly data is replicated. Sinks are not
	// supposed to modify these lists, thus no writers.
	//
	// Or.... assembly_rpc in general will not lock. It's only going to be
	// called by sinks anyway.
	return -1;
}

int assembly_rpc(asmid_t id, int vgpu, volatile struct cuda_packet *pkt)
{
	int err;
	struct assembly *assm = NULL;
	// doesn't involve main node
	// data paths should already be configured and set up
	err = pthread_mutex_lock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not lock internals\n");
	// search assembly list for the id, search for indicated vgpu, then RPC CUDA
	// packet to it.
	assm = find_assembly(id);
	if (unlikely(!assm)) {
		printd(DBG_ERROR, "could not locate assembly %lu\n", id);
		err = pthread_mutex_unlock(&internals->lock);
		if (err < 0)
			printd(DBG_ERROR, "Could not unlock internals\n");
		goto fail;
	}
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not unlock internals\n");
	// Execute calls. Some return data specific to the assembly, others can go
	// directly to NVIDIA's runtime.
	switch (pkt->method_id) {

		case CUDA_GET_DEVICE_COUNT:
			{
				int *devs = ((void*)pkt + pkt->args[0].argull);
				*devs = assm->num_gpus;
				break;
			}

		case CUDA_GET_DEVICE_PROPERTIES:
			{
				int dev;
				struct cudaDeviceProp *prop;
				dev = pkt->args[1].argll;
				prop = ((void*)pkt + pkt->args[0].argull);
				if (unlikely((unsigned int)dev >= assm->num_gpus)) {
					pkt->ret_ex_val.err = cudaErrorInvalidDevice;
					break;
				}
				memcpy(prop, &assm->mappings[dev].cudaDevProp,
						sizeof(struct cudaDeviceProp));
				break;
			}

		default:
			// Send to NVIDIA runtime.
			printd(DBG_ERROR, "Method %d not implemented yet.\n",
					pkt->method_id);
			goto fail;
			break;
	}
	return 0;
fail:
	return -1;
}

asmid_t assembly_request(const struct assembly_cap_hint *hint)
{
	int err;
	struct assembly *assm = NULL;
	if (!internals) {
		printd(DBG_ERROR, "assembly runtime not initialized\n");
		goto fail;
	}
	err = pthread_mutex_lock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not lock internals\n");
	assm = compose_assembly(hint);
	if (!assm) {
		printd(DBG_ERROR, "Could not compose assembly\n");
		pthread_mutex_unlock(&internals->lock);
		goto fail;
	}
	list_add(&assm->list, &internals->assembly_list);
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0)
		printd(DBG_ERROR, "Could not unlock internals\n");
	return assm->id;

fail:
	return INVALID_ASSEMBLY_ID;
}

int assembly_teardown(asmid_t id)
{
	// TODO send rpc to main to close the assembly if a minion, but we do not
	// unregister ourself with main
	return -1;
}

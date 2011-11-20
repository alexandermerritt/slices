/**
 * @file assembly.c
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
 *
 * FIXME Remove hardcoding of char arrays with size 255 everywhere.
 * FIXME Remove hardcoding of IP addresses everywhere, too
 *
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
 */

// System includes
#include <errno.h>
#include <ifaddrs.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// Project includes
#include <assembly.h>
#include <debug.h>
#include <method_id.h>
#include <util/list.h>

// Directory-immediate includes
#include "rpc.h"
#include "sinks.h"

/*-------------------------------------- INTERNAL STRUCTURES -----------------*/

enum vgpu_fixation
{
	VGPU_LOCAL = 0,
	VGPU_REMOTE
};

#define MAPPING_MAX_SET_TID 4

/**
 * State associating a vgpu with a pgpu.
 */
struct vgpu_mapping
{
	struct assembly *assm; //! Assembly holding this vgpu
	enum vgpu_fixation fixation;

	int vgpu_id; //! vgpu ID given to the application
	char vgpu_hostname[255]; // FIXME this will be different for all MPI ranks
	char vgpu_ip[255]; // FIXME this will be different for all MPI ranks

	int pgpu_id; //! physical gpu ID assigned by the local driver
	char pgpu_hostname[255];
	char pgpu_ip[255];
	struct cudaDeviceProp cudaDevProp;

	//! threads which have called setDevice on this vgpu
	pthread_t set_tids[MAPPING_MAX_SET_TID];
	int num_set_tids;
};

#define ASSEMBLY_MAX_MAPPINGS	24

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
	struct vgpu_mapping mappings[ASSEMBLY_MAX_MAPPINGS];
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
	char hostname[255];
	char ip[PARTICIPANT_MAX_NICS][255];
	char nic_name[PARTICIPANT_MAX_NICS][255];

	// GPU information
	int num_gpus;
	struct cudaDeviceProp dev_prop[PARTICIPANT_MAX_GPUS];
	int driverVersion, runtimeVersion;
};

struct main_state
{
	//! Used by main node to assign global assembly IDs
	asmid_t next_asmid;

	//! List of node_participant structures representing available nodes.
	struct list_head participants;
	
	// RPC thread state is contained inside rpc.c
	// Functions exist to determine if the thread is alive or not
};

struct minion_state
{
	//! State associated with an RPC connection to the MAIN node
	struct rpc_connection rpc_connection;
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

static int
init_internals(void)
{
	if (internals)
		return 0;
	internals = calloc(1, sizeof(*internals));
	if (!internals)
		return -ENOMEM;
	pthread_mutex_init(&internals->lock, NULL);
	INIT_LIST_HEAD(&internals->assembly_list);
	return 0;
}

/**
 * Fill in a participant structure with information about ourself. The caller
 * is responsible for setting the participant type and initializing the link
 * structure if necesary.
 *
 * @return	0 no issues
 * 			-ENETDOWN for errors obtaining network information
 * 			-ENODEV for errors with any CUDA Runtime or Driver API calls, or
 * 					if there are no GPUs detectable in the system
 */
static int
self_participant(struct node_participant *p)
{
	int err, exit_errno;
	struct ifaddrs *addrstruct = NULL, *ifa = NULL;
	void *addr = NULL; // actually some crazy sock type
	char addr_buffer[INET_ADDRSTRLEN];
	cudaError_t cuda_err;
	int idx = 0; // used for arrays in node_participant

	// Get our hostname.
	err = gethostname(p->hostname, 255);
	if (err < 0) {
		exit_errno = -ENETDOWN;
		goto fail;
	}

	// Figure out all our IP addresses.
	// http://stackoverflow.com/questions/212528/ (cont.)
	// 		linux-c-get-the-ip-address-of-local-computer
	err = getifaddrs(&addrstruct); // OS gives us a list of addresses
	if (err < 0) {
		exit_errno = -ENETDOWN;
		goto fail;
	}
	idx = 0;
	for (ifa = addrstruct; !ifa; ifa = ifa->ifa_next) { // iterate the list
		if (ifa->ifa_addr->sa_family == AF_INET) { // IPv4
			addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
			inet_ntop(AF_INET, addr, addr_buffer, INET_ADDRSTRLEN);
			snprintf(p->ip[idx], 255, "%s", ifa->ifa_name); // name of NIC
			snprintf(p->nic_name[idx], 255, "%s", addr_buffer); // NIC IP
			idx++;
			if (idx >= PARTICIPANT_MAX_NICS)
				break;
		}
	}
	freeifaddrs(addrstruct);

	// GPU count and their properties
	cuda_err = cudaGetDeviceCount(&p->num_gpus);
	if (cuda_err != CUDA_SUCCESS) {
		exit_errno = -ENODEV;
		goto fail;
	}
	if (p->num_gpus == 0) { // require nodes to have GPUs
		exit_errno = -ENODEV;
		goto fail;
	}
	for (idx = 0; idx < p->num_gpus; idx++) {
		cuda_err = cudaGetDeviceProperties(&p->dev_prop[idx], idx);
		if (cuda_err != CUDA_SUCCESS) {
			exit_errno = -ENODEV;
			goto fail;
		}
	}

	// Driver and runtime versions
	cuda_err = cudaDriverGetVersion(&p->driverVersion);
	if (cuda_err != CUDA_SUCCESS) {
		exit_errno = -ENODEV;
		goto fail;
	}
	cuda_err = cudaRuntimeGetVersion(&p->runtimeVersion);
	if (cuda_err != CUDA_SUCCESS) {
		exit_errno = -ENODEV;
		goto fail;
	}
	return 0;

fail:
	return exit_errno;
}

/**
 * Locate the assembly structure given the ID. Assumes locks governing the list
 * have been acquired by the caller.
 */
static struct assembly *
__find_assembly(asmid_t id)
{
	struct assembly *assm;
	list_for_each_entry(assm, &internals->assembly_list, link)
		if (assm->id == id)
			break;
	if (!assm || assm->id != id)
		assm = NULL;
	return assm;
}

/**
 * Locate the vgpu a thread has called cudaSetDevice on originally.  We assume
 * the caller holds any locks needed to protect the assembly or mapping state.
 * We also assume there exists only one association, as we stop once the thread
 * ID has been found.
 *
 * @param assm	The assembly to search
 * @param tid	The thread ID to search for in all vgpu mappings
 * @param idx	Optional output parameter containing the index into the
 * 				assocation state The caller should not consider its value if we
 * 				return NULL. If idx is NULL, it is ignored.
 * @return Pointer to the vgpu within the assembly if found, else NULL
 */
static struct vgpu_mapping *
get_thread_association(struct assembly *assm, pthread_t tid, int *idx)
{
	int found = 0; // 1 if the mapping is found, 0 otherwise
	struct vgpu_mapping *vgpu;
	int vgpu_id, tid_idx;
	// iterate over all vgpu mappings in assembly
	for (vgpu_id = 0; vgpu_id < assm->num_gpus && !found; vgpu_id++) {
		vgpu = &assm->mappings[vgpu_id];
		// iterate over tids already associated with this vgpu
		for (tid_idx = 0; tid_idx < vgpu->num_set_tids && !found; tid_idx++) {
			if (0 != pthread_equal(tid, vgpu->set_tids[tid_idx])) {
				found = 1;
				break;
			}
		}
	}
	if (!found)
		vgpu = NULL;
	else if (idx)
		*idx = tid_idx;
	return vgpu;
}

/**
 * Carry out the functionality of calling cudaSetDevice. Create an association
 * between the thread and the vgpu_id so that we can reference it later, and use
 * it to return queries to cudaGetDevice. We assume the caller holds any locks
 * needed to protect the assembly or mapping state.
 *
 * This function will ensure that no thread can map to more than one vgpu. In
 * other words, calling cudaSetDevice more than once will update the state to be
 * that of the latest invocation.
 *
 * @param assm		The assembly to modify
 * @param tid		The thread ID to create an association for
 * @param vgpu_id	The vgpu ID the specified thread wishes to use
 * @return The vgpu associated with the specified thread (should never be NULL).
 */
static struct vgpu_mapping *
set_thread_association(struct assembly *assm, pthread_t tid, int vgpu_id)
{
	struct vgpu_mapping *vgpu;
	int set_tid_idx;
	// First determine if the thread has previously called setDevice; we should
	// find an existing association to a vgpu
	vgpu = get_thread_association(assm, tid, &set_tid_idx);
	if (vgpu) {
		vgpu->set_tids[set_tid_idx] = 0; // FYI zero is not an invalid pthread_t
		vgpu->num_set_tids--;
	}
	// create the assocation
	vgpu = &assm->mappings[vgpu_id];
	vgpu->set_tids[vgpu->num_set_tids] = tid;
	vgpu->num_set_tids++;
	return vgpu;
}

/**
 * Figure out the composition of a new assembly given the hints, monitoring
 * state and currently existing assemblies. hostname is used to verify the host
 * has joined the assembly network (i.e. is in the participants list) and is
 * used to identify a new assembly's source.
 */
static struct assembly *
compose_assembly(const struct assembly_cap_hint *hint)
{
	struct node_participant *node = NULL;
	struct assembly *assm = calloc(1, sizeof(struct assembly));
	int vgpu_id;
	struct vgpu_mapping *vgpu;
	if (!assm) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&assm->link);
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
		printd(DBG_WARNING, "Hint asks for %d GPUs; providing %d\n",
				hint->num_gpus, ASSEMBLY_MAX_MAPPINGS);
		assm->num_gpus = ASSEMBLY_MAX_MAPPINGS;
	} else {
		assm->num_gpus = hint->num_gpus;
	}

	// find physical GPUs and set each mapping
	// FIXME we currently assume the main node can satisfy any assembly request.
	vgpu_id = 0;
	list_for_each_entry(node, &internals->n.main.participants, link) {
		if (node->type != NODE_TYPE_MAIN) // FIXME don't limit to main node
			continue;
		for ( ; vgpu_id < assm->num_gpus; vgpu_id++) {
			if (vgpu_id >= node->num_gpus)
				break; // skip to next node
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

	// FIXME Verify cuda/runtime versions on all nodes the assembly was mapped
	// to are equal. For now, just use the values from the first node in the
	// list.
	node = list_first_entry(&internals->n.main.participants,
			struct node_participant, link);
	assm->runtimeVersion = node->runtimeVersion;
	assm->driverVersion = node->driverVersion;

	return assm;

fail:
	if (assm)
		free(assm);
	return NULL;
}

static int
node_main_init(void)
{
	int err;
	struct node_participant *p;
	struct main_state *main_state;

	internals->type = NODE_TYPE_MAIN;
	main_state = &internals->n.main;
	main_state->next_asmid = 1UL;
	INIT_LIST_HEAD(&main_state->participants);

	// participant state
	p = calloc(1, sizeof(*p));
	if (!p)
		goto fail;
	err = self_participant(p);
	switch (err) {
		case 0: break;
		case -ENODEV:
			fprintf(stderr, "Error: no GPUs in node,"
						" or error with CUDA runtime/driver\n");
			goto fail;
		case -ENETDOWN:
			fprintf(stderr, "Error: cannot access network information\n");
			goto fail;
		default:
			goto fail;
	}
	p->type = NODE_TYPE_MAIN;
	INIT_LIST_HEAD(&p->link);

	// no need to lock list, as nobody else exists at this point
	list_add(&p->link, &internals->n.main.participants);

	err = rpc_enable();
	if (err < 0)
		goto fail;

	return 0;

fail:
	return -1;
}

static int
node_main_shutdown(void)
{
	int err, exit_errno = 0;
	struct node_participant *node_pos = NULL, *node_tmp = NULL;

	err = rpc_disable();
	if (err < 0) {
		exit_errno = -EPROTO;
		fprintf(stderr, "Error halting internal assembly RPC\n");
	}

	// Free participant list. We assume all minions have unregistered at this
	// point (meaning only one entry in the list).
	list_for_each_entry_safe(node_pos, node_tmp,
			&internals->n.main.participants, link) {
		if (node_pos->type == NODE_TYPE_MINION) { // oops...
			printd(DBG_ERROR, "minion still connected: TODO\n");
		}
		list_del(&node_pos->link);
		free(node_pos);
	}

	// We assume minions have shutdown their assemblies, leaving this list
	// empty.
	if (!list_empty(&internals->assembly_list)) {
		exit_errno = -EPROTO;
		printd(DBG_ERROR, "Assemblies still exist!\n");
		fprintf(stderr, "Assemblies still exist!\n");
	}

	free(internals);
	internals = NULL;

	return exit_errno;
}

static int
node_minion_init(const char *main_ip)
{
#if 0
	int err, exit_errno = -1;
	struct node_participant p;

	// minion state
	internals->type = NODE_TYPE_MINION;
	// FIXME rpc_init_conn internals->u.minion.conn

	// participant state
	p = calloc(1, sizeof(*p));
	if (!p)
		goto fail;
	err = self_participant(p);
	switch (err) {
		case 0: break;
		case -ENODEV:
			fprintf(stderr, "Error: no GPUs in node,"
						" or error with CUDA runtime/driver\n");
			goto fail;
		case -ENETDOWN:
			fprintf(stderr, "Error: cannot access network information\n");
			goto fail;
		default:
			goto fail;
	}
	p->type = NODE_TYPE_MINION;
	INIT_LIST_HEAD(&p->link);

#error rpc_send_join

	return 0;

fail:
	if (rpc_buffer)
		free(rpc_buffer);
	return exit_errno;
#endif
	return -1;
}

static int
node_minion_shutdown(void)
{
	// TODO Unregister with main
	fprintf(stderr, "Minion node not yet supported\n");
	return -1;
}

/*-------------------------------------- EXTERNAL FUNCTIONS ------------------*/

/*
 * These functions are callable outside this file, but are not part of the API.
 * They're used in rpc.c to support a distributed assembly runtime.
 */

// Return the assembly structure associated with the given ID.
struct assembly *
assembly_find(asmid_t id)
{
	struct assembly *assm;
	pthread_mutex_lock(&internals->lock);
	assm = __find_assembly(id);
	pthread_mutex_unlock(&internals->lock);
	return assm;
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int assembly_runtime_init(enum node_type type)
{
	int err;
	if (type >= NODE_TYPE_INVALID) {
		printd(DBG_ERROR, "invalid type: %u\n", type);
		goto fail;
	}
	err = init_internals();
	if (err < 0)
		goto fail;
	if (type == NODE_TYPE_MINION) {
		err = node_minion_init("10.0.0.1");
	} else if (type == NODE_TYPE_MAIN) {
		err = node_main_init();
	}
	if (err < 0)
		goto fail;
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
	if (err < 0) {
		printd(DBG_ERROR, "Could not lock internals\n");
		goto fail;
	}
	list_for_each_entry(assm, &internals->assembly_list, link) {
		if (assm->id == id) {
			num_gpus = assm->num_gpus;
			break;
		}
	}
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not unlock internals\n");
	}
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

/**
 * Execute an RPC (CUDA for now) in the runtime. Use of the vgpu identifier is
 * only needed for remote vgpu mappings, to determine which network connection
 * state to send the RPC on.
 *
 * FIXME Clean up the layout of the code in this function.
 *
 * IMPORTANT NOTE
 * If the process in which this function is called from contains multiple
 * threads invoking this function, those threads will need to ensure that they
 * appropriately switch GPUs (cudaSetDevice) IF such a thread is to multiplex
 * between devices (not necessarily the best thing to do anyway). In other
 * words, local vgpu mappings ignore the vgpu identifier in this function, and
 * directly pass the RPC to the NVIDIA runtime.  Ensuring the call goes to the
 * appropriate local GPU is the responsibility of the caller. There are no
 * threads behind this interface to ensure this; each call into this function
 * assumes the CUDA driver context of that thread directly.
 *
 * TODO Will need to duplicate all register calls to each remote node within the
 * assembly.
 */
int assembly_rpc(asmid_t id, int vgpu_id, volatile struct cuda_packet *pkt)
{
	int err;
	struct assembly *assm = NULL;
	// doesn't involve main node
	// data paths should already be configured and set up
	err = pthread_mutex_lock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not lock internals\n");
		goto fail;
	}
	// search assembly list for the id, search for indicated vgpu, then RPC CUDA
	// packet to it.
	assm = __find_assembly(id);
	if (!assm) {
		printd(DBG_ERROR, "could not locate assembly %lu\n", id);
		err = pthread_mutex_unlock(&internals->lock);
		if (err < 0) {
			printd(DBG_ERROR, "Could not unlock internals\n");
		}
		goto fail;
	}
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not unlock internals\n");
		goto fail;
	}
	// Execute calls. Some return data specific to the assembly, others can go
	// directly to NVIDIA's runtime.
	switch (pkt->method_id) {

		case CUDA_GET_DEVICE:
			{
				int *dev = (int*)((uintptr_t)pkt + pkt->args[0].argull);
				struct vgpu_mapping *vgpu;
				// A thread may or may not have previously called cudaSetDevice.
				// If it has not, assign it to vgpu 0 and return that (this
				// models the behavior of the CUDA runtime). If it has, then
				// simply return what that association is.
				vgpu = get_thread_association(assm, pkt->thr_id, NULL);
				if (!vgpu)
					vgpu = set_thread_association(assm, pkt->thr_id, 0);
				*dev = vgpu->vgpu_id;
				printd(DBG_DEBUG, "getDev=%d\n", *dev);
			}
			break;

		case CUDA_GET_DEVICE_COUNT:
			{
				int *devs = (int*)((uintptr_t)pkt + pkt->args[0].argull);
				*devs = assm->num_gpus;
				printd(DBG_DEBUG, "num devices=%d\n", *devs);
			}
			break;

		case CUDA_GET_DEVICE_PROPERTIES:
			{
				struct cudaDeviceProp *prop;
				int dev;
				prop = (struct cudaDeviceProp*)
							((uintptr_t)pkt + pkt->args[0].argull);
				dev = pkt->args[1].argll;
				if (dev < 0 || dev >= assm->num_gpus) {
					printd(DBG_WARNING, "invalid dev id %d\n", dev);
					pkt->ret_ex_val.err = cudaErrorInvalidDevice;
					break;
				}
				memcpy(prop, &assm->mappings[dev].cudaDevProp,
						sizeof(struct cudaDeviceProp));
				printd(DBG_DEBUG, "name=%s\n", prop->name);
			}
			break;

		case CUDA_DRIVER_GET_VERSION:
			{
				int *ver = (int*)((uintptr_t)pkt + pkt->args[0].argull);
				*ver = assm->driverVersion;
				printd(DBG_DEBUG, "driver ver=%d\n", *ver);
			}
			break;

		case CUDA_RUNTIME_GET_VERSION:
			{
				int *ver = (int*)((uintptr_t)pkt + pkt->args[0].argull);
				*ver = assm->runtimeVersion;
				printd(DBG_DEBUG, "runtime ver=%d\n", *ver);
			}
			break;

		case CUDA_SET_DEVICE:
			{
				int devid = pkt->args[0].argll;
				struct vgpu_mapping *vgpu;
				if (devid >= assm->num_gpus) { // failure at application
					pkt->ret_ex_val.err = cudaErrorInvalidDevice;
					break; // exit switch: don't pass to NVIDIA runtime
				} else {
					// Create association, then modify vgpu_id in pkt argument
					// to be the physical device ID that the vgpu ID represents
					vgpu = set_thread_association(assm, pkt->thr_id, devid);
					pkt->args[0].argll = vgpu->pgpu_id;
				}
				printd(DBG_DEBUG, "thread=%lu vgpu=%d pgpu=%d\n",
						pkt->thr_id, vgpu->vgpu_id, vgpu->pgpu_id);
			}
			// Let setDevice fall through to NVIDIA runtime, as driver also
			// needs to maintain a mapping for real thread contexts.

		default: // Send to NVIDIA runtime.
			{
				struct vgpu_mapping *vgpu = &assm->mappings[vgpu_id];
				if (vgpu->fixation == VGPU_REMOTE) {
					// TODO Implement this path when the networking code becomes
					// available.
					printd(DBG_ERROR, "Remote vgpus not yet supported\n");
					goto fail;
				} else {
					err = nv_exec_pkt(pkt);
					if (err < 0) {
						printd(DBG_ERROR,
								"Failed executing packet at NV runtime\n");
						goto fail;
					}
				} // vgpu fixation
			} // switch default
			break;
	} // switch method_id
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
	if (err < 0) {
		printd(DBG_ERROR, "Could not lock internals\n");
		goto fail;
	}
	assm = compose_assembly(hint);
	if (!assm) {
		printd(DBG_ERROR, "Could not compose assembly\n");
		pthread_mutex_unlock(&internals->lock);
		goto fail;
	}
	list_add(&assm->link, &internals->assembly_list);
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not unlock internals\n");
	}
	return assm->id;

fail:
	return INVALID_ASSEMBLY_ID;
}

int assembly_teardown(asmid_t id)
{
	// TODO send rpc to main to close the assembly if a minion, but we do not
	// unregister ourself with main
	int err;
	struct assembly *assm = NULL;
	if (!internals) {
		printd(DBG_ERROR, "assembly runtime not initialized\n");
		goto fail;
	}
	err = pthread_mutex_lock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not lock internals\n");
		goto fail;
	}
	assm = __find_assembly(id);
	if (!assm) {
		printd(DBG_ERROR, "Invalid assembly ID %lu\n", id);
		goto fail;
	}
	list_del(&assm->link);
	free(assm);
	err = pthread_mutex_unlock(&internals->lock);
	if (err < 0) {
		printd(DBG_ERROR, "Could not unlock internals\n");
		goto fail;
	}
	return 0;

fail:
	return -1;
}

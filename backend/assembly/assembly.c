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
#include "types.h"

/*-------------------------------------- INTERNAL STRUCTURES -----------------*/

struct main_state
{
	//! Used by main node to assign global assembly IDs
	asmid_t next_asmid;

	//! List of node_participant structures representing available nodes.
	struct list_head participants;
	pthread_mutex_t plock; //! participant list lock
	
	// RPC thread state is contained inside rpc.c
	// Functions exist to determine if the thread is alive or not
};

struct minion_state
{
	//! State associated with an RPC connection to the MAIN node
	struct rpc_connection rpc_conn;
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
	for (ifa = addrstruct; ifa; ifa = ifa->ifa_next) { // iterate the list
		if (ifa->ifa_addr->sa_family == AF_INET) { // IPv4
			addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
			inet_ntop(AF_INET, addr, addr_buffer, INET_ADDRSTRLEN);
			snprintf(p->nic_name[idx], 255, "%s", ifa->ifa_name); // name of NIC
			snprintf(p->ip[idx], 255, "%s", addr_buffer); // NIC IP
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
	} else if (hint->num_gpus > MAX_VGPUS) {
		printd(DBG_WARNING, "Hint asks for %d GPUs; providing %d\n",
				hint->num_gpus, MAX_VGPUS);
		assm->num_gpus = MAX_VGPUS;
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

// forward declaration
int add_participant(struct node_participant *p);

static int
node_main_init(void)
{
	int err;
	struct node_participant *p = NULL;
	struct main_state *state;
	int iface;

	internals->type = NODE_TYPE_MAIN;
	state = &internals->n.main;
	state->next_asmid = 1UL;
	INIT_LIST_HEAD(&state->participants);

	// Create a participant describing ourself
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

	// Print out which interfaces we discovered for the user.
	printf("Listening on interfaces:");
	for (iface = 0; iface < PARTICIPANT_MAX_NICS; iface++) {
		if (*(p->nic_name[iface]) == '\0')
			break;
		printf(" (%s:%s)", p->nic_name[iface], p->ip[iface]);
	}
	printf("\n");

	err = rpc_enable();
	if (err < 0)
		goto fail;

	err = add_participant(p);
	if (err < 0) { // this is never expected
		goto fail;
	}

	return 0;

fail:
	rpc_disable();
	if (p) free(p);
	return -1;
}

static int
node_main_shutdown(void)
{
	int exit_errno = 0;
	struct node_participant *node_pos = NULL, *node_tmp = NULL;

	rpc_disable();

	// Free participant list. We assume all minions have unregistered at this
	// point (meaning only one entry in the list).
	// FIXME Still need to do this
	list_for_each_entry_safe(node_pos, node_tmp,
			&internals->n.main.participants, link) {
		if (node_pos->type == NODE_TYPE_MINION) { // oops...
			printd(DBG_ERROR, "Remote host %s still connected!\n",
					node_pos->hostname);
		}
		list_del(&node_pos->link);
		free(node_pos);
	}

	// We assume minions have shutdown their assemblies, leaving this list
	// empty.
	// FIXME Still need to do this
	if (!list_empty(&internals->assembly_list)) {
		exit_errno = -EPROTO;
		printd(DBG_ERROR, "Assemblies still exist!\n");
		fprintf(stderr, "Assemblies still exist!\n");
	}

	free(internals);
	internals = NULL;

	return exit_errno;
}

/**
 * Initialize a node to be of type minion. No participant list is maintained. An
 * assembly list is maintained, but only for those for which we've received
 * requests.  Initialization includes creating populating a participant
 * structure, and sending it to the main node. Departure from the assembly
 * network, requesting and tearing down assemblies require command RPCs with the
 * main node.
 */
static int
node_minion_init(const char *main_ip)
{
	int err, exit_errno;
	struct node_participant *p = NULL;
	struct minion_state *state;

	BUG(!internals);
	internals->type = NODE_TYPE_MINION;
	state = &internals->n.minion;

	// Connect to the main node
	err = rpc_init_conn(&state->rpc_conn);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	err = rpc_connect(&state->rpc_conn, main_ip);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}

	// Create a participant describing ourself
	p = calloc(1, sizeof(*p));
	if (!p) {
		exit_errno = -ENOMEM;
		goto fail;
	}
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

	// Send our registration, equivalent to add_participant in main_init
	err = rpc_send_join(&state->rpc_conn, p);
	if (err < 0) {
		// errors may include network death, main out of memory, message
		// corruption, duplicate participant entry etc
		exit_errno = err;
		goto fail;
	}

	free(p);
	return 0;

fail:
	rpc_close(&state->rpc_conn);
	if (p) free(p);
	return exit_errno;
}

static int
node_minion_shutdown(void)
{
	int err, exit_errno;
	struct minion_state *state = NULL;

	BUG(!internals);
	state = &internals->n.minion;

	// XXX FIXME If there are still assemblies, remove them one-by-one.

	// Stop on the first error as all of these involve network communication. In
	// the worst case, Linux will close all our file descriptors for us when we
	// exit or crash. The MAIN node is able to detect this and will attempt to
	// clean up for us, anyway.
	err = rpc_send_leave(&state->rpc_conn);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	err = rpc_close(&state->rpc_conn);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	err = rpc_tini_conn(&state->rpc_conn);
	if (err < 0) {
		exit_errno = err;
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

//! Same as get_participant, but without locking. Not accessible externally.
static struct node_participant *
__get_participant(const char *hostname)
{
	struct node_participant *iter = NULL;
	list_for_each_entry(iter, &internals->n.main.participants, link)
		if (strncmp(hostname, iter->hostname, HOST_LEN) == 0)
			break;
	if (iter && strncmp(hostname, iter->hostname, HOST_LEN) == 0)
		return iter;
	return NULL;
}

/*-------------------------------------- EXTERNAL FUNCTIONS ------------------*/

/*
 * These functions are callable outside this file, but are not part of the API.
 * They're used both within this file and in rpc.c to support a distributed
 * assembly runtime.
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

/**
 * Look for a participant in the list with the given hostname. Return true if
 * yes and false if no.
 */
bool
participant_exists(const char *hostname)
{
	struct node_participant *exists = NULL;
	pthread_mutex_lock(&internals->n.main.plock);
	exists = __get_participant(hostname);
	pthread_mutex_unlock(&internals->n.main.plock);
	return (exists != NULL);
}

/**
 * Add a participant structure to the list the MAIN node maintains. MINION nodes
 * do not have such a list, as it is only used to map assemblies. Assumes the
 * list has been initialized and the argument is valid.  For all error
 * conditions returned, the participant is NOT added to the list.
 *
 * @return	-EEXIST if a participant with the same hostname as one in the list
 *			is given.
 */
int
add_participant(struct node_participant *p)
{
	struct node_participant *existing = NULL;
	pthread_mutex_lock(&internals->n.main.plock);
	existing = __get_participant(p->hostname);
	if (existing) {
		pthread_mutex_unlock(&internals->n.main.plock);
		return -EEXIST;
	}
	list_add(&p->link, &internals->n.main.participants);
	pthread_mutex_unlock(&internals->n.main.plock);
	return 0;
}

/**
 * Remove a participant structure from the list the MAIN node maintains. It uses
 * the hostname for comparison only. Assumes the list has been initialized and
 * the argument is valid.
 *
 * @param	removed		If hostname matches a participant in the list, this will
 * 						point to the structure removed from the list matching
 * 						that hostname.
 * @return	-EINVAL if no participant in the list has a hostname matching the
 * 			given hostname.
 */
int
rm_participant(const char *hostname, struct node_participant **removed)
{
	struct node_participant *existing = NULL;
	pthread_mutex_lock(&internals->n.main.plock);
	existing = __get_participant(hostname);
	if (!existing) {
		pthread_mutex_unlock(&internals->n.main.plock);
		return -EINVAL;
	}
	list_del(&existing->link);
	*removed = existing; // pointer copy
	pthread_mutex_unlock(&internals->n.main.plock);
	return 0;
}


/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int assembly_runtime_init(enum node_type type, const char *main_ip)
{
	int err;
	err = init_internals();
	if (err < 0)
		goto fail;
	if (type == NODE_TYPE_MINION) {
		err = node_minion_init(main_ip);
	} else if (type == NODE_TYPE_MAIN) {
		err = node_main_init();
	} else {
		goto fail;
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

// forward declaration
extern int nv_exec_pkt(volatile struct cuda_packet *pkt);

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

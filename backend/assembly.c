/**
 * @file assembly.c
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
 *
 * Restriction: DO NOT call into the CUDA Runtime API within this file, use
 * hack_getCudaInformation to obtain CUDA information. It forks a child to make
 * the calls, leaving the Runtime API state uninitialized here.
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <assembly.h>
#include <debug.h>
#include <method_id.h>
#include <sinks.h>
#include <util/compiler.h>
#include <util/list.h>

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
	char vgpu_hostname[255];
	char vgpu_ip[255];

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

/**
 * State describing a node that has registered to participate in the assembly
 * runtime. Minions send RPCs to the main node to register. Not using any
 * pointers in this structure as that eliminates the need to serialize it when
 * sending it as part of an RPC.
 */
struct node_participant
{
	struct list_head link;
	char hostname[255];
	char ip[255];
	int num_gpus;
	struct cudaDeviceProp dev_prop[PARTICIPANT_MAX_GPUS];
	int driverVersion, runtimeVersion;
	enum node_type type;
};

struct main_state
{
	//! Used by main node to assign global assembly IDs
	asmid_t next_asmid;

	//! List of node_participant structures representing available nodes.
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
static struct assembly*
__find_assembly(asmid_t id)
{
	struct assembly *assm = NULL;
	list_for_each_entry(assm, &internals->assembly_list, link)
		if (assm->id == id)
			break;
	if (unlikely(!assm || assm->id != id)) {
		printd(DBG_ERROR, "unknown assembly id %lu\n", id);
	}
	return assm;
}

/**
 * This is a hack. Child processes forked from parent processes which have made
 * calls into the CUDA Runtime API are not able to call into the Runtime API
 * themselves.  Functions return an error indicating "no device available". This
 * function allows the caller to invoke certain CUDA Runtime functions without
 * initializing any context within its address space.
 *
 * This problem has been confirmed with CUDA v3.2 and is visible all over the
 * NVIDIA forums. A solution was needed for the assembly runtime, as the first
 * process executed within the backend needs to probe the system for available
 * GPUs as part of registering in the runtime. When apps 'connect' to the
 * runtime, this process will fork a child 'sink' process to handle the
 * requests. If the first makes calls into the Runtime API, the latter will fail
 * to handle CUDA RPCs to local GPUs.
 *
 * This function assumes a machine has no more than 32 GPUs contained within it.
 *
 * @param node		Node participant data structure that will be updated with
 * 					regards to the number of GPUs present, all their device
 * 					properties and the driver and runtime version numbers.
 *
 * @return	0 okay, -1 not okay
 */
static int hack_getCudaInformation(struct node_participant *node)
{
	int err;
	struct cudaDeviceProp *props = node->dev_prop;
	int *num_gpus = &node->num_gpus;
	int *drv_ver = &node->driverVersion;
	int *cuda_ver = &node->runtimeVersion;

	// Determine the number of GPUs. The CUDA Driver API doesn't act as screwy
	// as the Runtime API, so this is safe to do.
	CUresult cu_err;
	cu_err = cuInit(0);
	if (cu_err != CUDA_SUCCESS)
		goto fail;
	cu_err = cuDeviceGetCount(num_gpus);
	if (cu_err != CUDA_SUCCESS)
		goto fail;

	// Create a shared memory region and fork a child process to do the work.
	// The child will write the array of cudaDeviceProp structs to this region
	// which the parent (the caller of the function) can then read in. The child
	// has a separate instance of the CUDA Runtime library with which it can
	// make function calls, unaffecting the parent.

	struct { // Layout of the shared memory region.
		int _drv_ver;
		int _cuda_ver;
		struct cudaDeviceProp _props[32]; // here's that assumption
	} *format;
	void *shm = NULL;
	size_t shm_size = sizeof(*format);
	// mmaping an 'anonymous' region doesn't require us to open a file
	shm = mmap(NULL, shm_size, PROT_READ | PROT_WRITE,
			MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (shm == MAP_FAILED)
		goto fail;
	format = shm;

	pid_t pid;
	pid = fork();
	if (pid == 0) { // Child process does all the CUDA Runtime calls.
		cudaError_t cuda_err;
		int dev;
		cuda_err = cudaDriverGetVersion(&format->_drv_ver);
		if (cuda_err != cudaSuccess)
			exit(-1);
		cuda_err = cudaRuntimeGetVersion(&format->_cuda_ver);
		if (cuda_err != cudaSuccess)
			exit(-1);
		for (dev = 0; dev < *num_gpus; dev++) {
			cuda_err = cudaGetDeviceProperties(&format->_props[dev], dev);
			if (cuda_err != cudaSuccess)
				exit(-1);
		}
		exit(0);
	} else if (pid > 0) { // Parent process
		int status = 0;
		pid_t _pid;
		_pid = waitpid(pid, &status, 0); // Wait for the child to exit.
		if (_pid < 0 || (_pid != pid))
			goto fail;
	} else { // fork() failure
		goto fail;
	}

	// Copy results to user-supplied arguments.
	memcpy(props, format->_props, (*num_gpus * sizeof(struct cudaDeviceProp)));
	*drv_ver = format->_drv_ver;
	*cuda_ver = format->_cuda_ver;
	err = munmap(shm, shm_size);
	shm = NULL;
	if (err != 0) goto fail;

	return 0;

fail:
	if (shm)
		munmap(shm, shm_size); // What if this also fails??
	shm = NULL;
	return -1;
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
		printd(DBG_WARNING, "Hint asks for %u GPUs; providing %u\n",
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
	node = list_first_entry(&internals->n.main.participants, struct node_participant, link);
	assm->runtimeVersion = node->runtimeVersion;
	assm->driverVersion = node->driverVersion;

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
	int err;

	internals->n.main.next_asmid = 1UL;
	INIT_LIST_HEAD(&internals->n.main.participants);

	p = calloc(1, sizeof(struct node_participant));
	if (!p) {
		printd(DBG_ERROR, "out of memory\n");
		fprintf(stderr, "out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&p->link);
	
	// FIXME don't hardcode, figure out at runtime
	strcpy(p->ip, "10.0.0.1");
	err = hack_getCudaInformation(p);
	if (err < 0)
		goto fail;
	printd(DBG_DEBUG, "node: gpus=%d drv ver=%d cuda ver=%d\n",
			p->num_gpus, p->driverVersion, p->runtimeVersion);
	strcpy(p->hostname, "ifrit"); // just for identification, not routing
	p->type = NODE_TYPE_MAIN;

	// no need to lock list, as nobody else exists at this point
	list_add(&p->link, &internals->n.main.participants);

	// TODO Spawn RPC thread.

	return 0;
fail:
	return -1;
}

static int node_main_shutdown(void)
{
	struct node_participant *node_pos = NULL, *node_tmp = NULL;

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
		printd(DBG_ERROR, "Assemblies still exist!\n");
		fprintf(stderr, "Assemblies still exist!\n");
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
	if (unlikely(!assm)) {
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
				int *dev = ((void*)pkt + pkt->args[0].argull);
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
				int *devs = ((void*)pkt + pkt->args[0].argull);
				*devs = assm->num_gpus;
				printd(DBG_DEBUG, "num devices=%d\n", *devs);
			}
			break;

		case CUDA_GET_DEVICE_PROPERTIES:
			{
				struct cudaDeviceProp *prop;
				int dev;
				prop = ((void*)pkt + pkt->args[0].argull);
				dev = pkt->args[1].argll;
				if (unlikely(dev < 0 || dev >= assm->num_gpus)) {
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
				int *ver = ((void*)pkt + pkt->args[0].argull);
				*ver = assm->driverVersion;
				printd(DBG_DEBUG, "driver ver=%d\n", *ver);
			}
			break;

		case CUDA_RUNTIME_GET_VERSION:
			{
				int *ver = ((void*)pkt + pkt->args[0].argull);
				*ver = assm->runtimeVersion;
				printd(DBG_DEBUG, "runtime ver=%d\n", *ver);
			}
			break;

		case CUDA_SET_DEVICE:
			{
				int id = pkt->args[0].argll;
				struct vgpu_mapping *vgpu;
				if (id >= assm->num_gpus) { // failure at application
					pkt->ret_ex_val.err = cudaErrorInvalidDevice;
					break; // exit switch: don't pass to NVIDIA runtime
				} else {
					// Create association, then modify vgpu_id in pkt argument to be
					// the physical device ID that the vgpu ID represents
					vgpu = set_thread_association(assm, pkt->thr_id, id);
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

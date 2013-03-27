/**
 * @file assembly.c
 * @date 2011-10-23
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief
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
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// Project includes
#include <assembly.h>
#include <cuda/fatcubininfo.h>
#include <cuda/marshal.h>
#include <cuda/method_id.h>
#include <debug.h>
#include <util/list.h>
#include <util/timer.h>

// Directory-immediate includes
#include "remote.h"
#include "rpc.h"
#include "types.h"
#include "internals.h"

/*-------------------------------------- INTERNAL STATE ----------------------*/

/* If no assembly hint is specified when an application process joins the
 * system, this is the hint used to construct an assembly.
 */
struct assembly_hint assembly_default_hint =
{
    .num_gpus = 1, /* per process */
#if defined(NIC_SDP)
    .nic_type = HINT_USE_IB,
#elif defined(NIC_ETHERNET)
    .nic_type = HINT_USE_ETH,
#else
#error NIC_* not defined
#endif
    .mpi_group = 0, /* := unused, not an MPI app */
    .policy = HINT_ENUM_POLICY_LOCALFIRST,
    .batch_size = CUDA_BATCH_MAX
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

struct internals_state *internals = NULL;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

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
	internals->rsink_pid = -1;
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

	memset(addr_buffer, 0, INET_ADDRSTRLEN);

	// Get our hostname.
	err = gethostname(p->hostname, HOST_LEN);
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
	p->num_nics = 0;
	for (ifa = addrstruct; ifa; ifa = ifa->ifa_next) { // iterate the list
		if (ifa->ifa_addr->sa_family == AF_INET) { // IPv4
			addr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
			inet_ntop(AF_INET, addr, addr_buffer, INET_ADDRSTRLEN);
			if (strncmp(ifa->ifa_name, "lo", 3) == 0)
				continue; // skip 127.0.0.1
			snprintf(p->nic_name[idx], HOST_LEN, "%s", ifa->ifa_name); // name of NIC
			snprintf(p->ip[idx], HOST_LEN, "%s", addr_buffer); // NIC IP
			printd(DBG_INFO, "%s has %s with %s\n",
					p->hostname, p->nic_name[idx], p->ip[idx]);
			p->num_nics++;
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
	bool found = false;
	struct assembly *assm;
	list_for_each_entry(assm, &internals->assembly_list, link) {
		if (assm->id == id) {
			found = true;
			break;
		}
	}
	if (!found) {
		assm = NULL;
	}
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

// forward declaration within assemble.c
struct assembly* __do_compose_assembly(const struct assembly_hint*, const char*,
		const struct list_head*, const struct list_head*);

/**
 * Figure out the composition of a new assembly given the hints, monitoring
 * state and currently existing assemblies. hostname is used to verify the host
 * has joined the assembly network (i.e. is in the participants list) and is
 * used to identify a new assembly's source.
 *
 * This function assumes the caller holds any required locks.
 */
static struct assembly *
__compose_assembly(const struct assembly_hint *hint, const char *hostname)
{
	struct assembly *assm = 
		__do_compose_assembly(hint, hostname,
				&internals->n.main.participants, &internals->assembly_list);
	if (!assm)
		return NULL;

	INIT_LIST_HEAD(&assm->link);
	assm->id = NEXT_ASMID;

	assm->cubins = malloc(sizeof(struct fatcubins));
	if (!assm->cubins) {
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	cubins_init(assm->cubins);

	assm->mapped = false;
	memcpy(&assm->hint, hint, sizeof(*hint));

	return assm;
fail:
	if (assm)
		free(assm);
	return NULL;
}

/* just decrement the pGPU counters for now */
/* internals lock should be held */
/* TODO maybe this code should be in assemble.c? */
static void
__decompose_assembly(struct assembly *assm)
{
    int vgpu_id;
	struct node_participant *node = NULL;
    struct vgpu_mapping *vgpu = NULL;
    for (vgpu_id = 0; vgpu_id < assm->num_gpus; vgpu_id++) {
        vgpu = &assm->mappings[vgpu_id];
        list_for_each_entry(node, &internals->n.main.participants, link)
            /* find node vgpu maps to */
            if (str_eq(node->hostname, vgpu->hostname, HOST_LEN))
                node->gpu_mapped[vgpu->pgpu_id]--;
    }
}

extern int start_remote_sink(void);

static int
remote_enable(void)
{
    return start_remote_sink();
#if 0
    if (0 != access("remotesink", R_OK)) {
        fprintf(stderr, "> remotesink not found in current directory\n");
        return -1;
    }

	BUG(!internals);
	pthread_mutex_lock(&internals->lock);
	if (internals->rsink_pid > 0) {
		pthread_mutex_unlock(&internals->lock);
		return -EEXIST;
	}
	pid = fork();

	// Error
	if (pid < 0) {
		pthread_mutex_unlock(&internals->lock);
		return -(errno);
	}

	// Parent; exit now
	if (pid > 0) {
		internals->rsink_kill_sig = REMOTE_TERM_SIG;
		internals->rsink_pid = pid;
		pthread_mutex_unlock(&internals->lock);
		return 0;
	}

	// We're the child. Exec the remote sink.
	err = execl("remotesink", REMOTE_EXEC_NAME, NULL);
	_exit(-1); // execl only returns on error
#endif
}

#if 0
static int
remote_disable(void)
{
	int err;
	BUG(!internals);
	pthread_mutex_lock(&internals->lock);
	if (internals->rsink_pid > 0) {
		err = kill(internals->rsink_pid, internals->rsink_kill_sig);
		if (err < 0) {
			pthread_mutex_lock(&internals->lock);
			return -(errno);
		}
		int status;
		err = waitpid(internals->rsink_pid, &status, 0);
		if (err < 0) {
			pthread_mutex_lock(&internals->lock);
			return -(errno);
		}
		if (!WIFEXITED(status)) {
			printd(DBG_DEBUG, "remotesink exited abnormally with code %d\n",
                    WEXITSTATUS(status));
		}
		internals->rsink_pid = -1;
	}
	pthread_mutex_unlock(&internals->lock);
	return 0;
}
#endif

// forward declaration
int add_participant(struct node_participant *p);

static int
node_main_init(void)
{
	int err;
	struct node_participant *p = NULL;
	struct main_state *state;

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

	err = add_participant(p);
	if (err < 0) { // this is never expected
		goto fail;
	}

	// turn on distributed assembly capability
	err = rpc_enable();
	if (err < 0)
		goto fail;

	// enable (remote) vgpus to connect to us
	err = remote_enable();
	if (err < 0) {
		goto fail;
	}

	return 0;

fail:
	rpc_disable();
	//remote_disable();
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

#if 0
	err = remote_disable();
	if (err < 0) {
		fprintf(stderr, "Could not shutdown remotesink PID %d\n",
				internals->rsink_pid);
	}
#endif

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
	int err = 0, exit_errno = 0;
	struct node_participant *p = NULL;
	struct minion_state *state;

	BUG(!internals);
	internals->type = NODE_TYPE_MINION;
	state = &internals->n.minion;

	err = remote_enable();
	if (err < 0) {
        return -1;
	}

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
			BUG(1);
			break;
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
	p = NULL;

	return 0;

fail:
	rpc_close(&state->rpc_conn);
	if (p) free(p);
    //remote_disable();
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
#if 0
	err = remote_disable();
	if (err < 0) {
		fprintf(stderr, "Could not shutdown remotesink PID %d\n",
				internals->rsink_pid);
	}
#endif
	return 0;
fail:
	return exit_errno;
}

static int
node_mapper_shutdown(void)
{
	struct assembly *assm = NULL, *tmp = NULL;
	// Remove the imported assemblies this mapper added (we actually only expect
	// each mapper to import one assembly). A parent process will be responsible
	// for calling assembly_request and assembly_teardown. Between these this
	// parent will export assemblies that mappers import. So here we only free
	// the state.
	pthread_mutex_lock(&internals->lock);
	list_for_each_entry_safe(assm, tmp, &internals->assembly_list, link) {
		list_del(&assm->link);
		free(assm);
	}
	pthread_mutex_unlock(&internals->lock);
	return 0;
}

//! Same as get_participant, but without locking. Not accessible externally.
static struct node_participant *
__get_participant(const char *hostname)
{
	struct node_participant *iter = NULL;
	list_for_each_entry(iter, &internals->n.main.participants, link)
		if (strncmp(hostname, iter->hostname, HOST_LEN) == 0)
			return iter;
	return NULL;
}

static int
__do_read_hint(const char *path, struct assembly_hint *hint)
{
    int err;
    const int linesz = 256;
    FILE *fptr = NULL;
    char line[linesz], *field, *val, *saveptr;

    memset(hint, 0, sizeof(*hint));
    memset(line, 0, linesz * sizeof(*line));

    err = access(path, R_OK);
    if (err != 0) {
        printd(DBG_INFO, "can't access %s for reading\n", path);
        goto fail;
    }

    fptr = fopen(path, "r");
    if (!fptr) {
        printd(DBG_INFO, "can't open %s\n", path);
        goto fail;
    }

    if (NULL == fgets(line, linesz, fptr)) {
        printd(DBG_INFO, "emtpy file %s, ignoring\n", path);
        return 0;
    }

    do {
        if (HINT_COMMENT_CHAR == *line)
            continue;

        field = strtok_r(line, HINT_EQUAL_DELIM, &saveptr);
        if (!field)
            goto fail_malformed; /* line has no = */
        val = strtok_r(NULL, HINT_EQUAL_DELIM, &saveptr);
        if (!val)
            goto fail_malformed; /* blabla= */

        if (strcmp(field, HINT_FIELD_MPI) == 0) {
            hint->mpi_group = atoi(val);
        } else if (strcmp(field, HINT_FIELD_POLICY) == 0) {
            hint->policy = assembly_hint_policy2enum(val);
        } else if (strcmp(field, HINT_FIELD_BATCHSIZE) == 0) {
            hint->batch_size = atoi(val);
        } else if (strcmp(field, HINT_FIELD_NIC) == 0) {
            /* TODO */
        /* TODO read num_vgpus */
        } else
            fprintf(stderr, "> hint file: unknown field '%s'\n", field);
    } while(NULL != fgets(line, linesz, fptr));

    /* fix up the fields */
    if (hint->policy == HINT_ENUM_POLICY_INVALID) {
        fprintf(stderr, "> Policy value invalid\n");
        goto fail_malformed;
    }

    if (hint->batch_size == 0)
        hint->batch_size = CUDA_BATCH_MAX;

    hint->num_gpus = 1; /* TODO don't do this once we read in num_vgpus */

#if defined(NIC_SDP)
    hint->nic_type = HINT_USE_IB;
#elif defined(NIC_ETHERNET)
    hint->nic_type = HINT_USE_ETH;
#else
#error NIC_* not defined
#endif

    fclose(fptr);
    return 0;

fail_malformed:
    fprintf(stderr, "> Malformed hint file, ignoring\n");
    memset(hint, 0, sizeof(*hint));

fail:
    if (fptr)
        fclose(fptr);
    return -1;
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
	printd(DBG_INFO, "%s type %s %d gpus\n",
			p->hostname, node_type_str(p->type), p->num_gpus);
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

/**
 * Simple function to acquire an assembly.
 *
 * @param	hint		The hint used to compose an assembly
 * @param	hostname	Who is requesting it.
 */
struct assembly *
request_assembly(const struct assembly_hint *hint, const char *hostname)
{
	struct assembly *assm;
	pthread_mutex_lock(&internals->lock);
	assm = __compose_assembly(hint, hostname);
	if (assm)
		list_add(&assm->link, &internals->assembly_list);
	pthread_mutex_unlock(&internals->lock);
	return assm;
}

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int assembly_runtime_init(enum node_type type, const char *main_ip)
{
	int err;
	err = init_internals();
	if (err < 0)
		goto fail;
	if (type == NODE_TYPE_MINION)
		err = node_minion_init(main_ip);
	else if (type == NODE_TYPE_MAIN)
		err = node_main_init();
    else if (type == NODE_TYPE_MAPPER) {
        internals->type = NODE_TYPE_MAPPER;
    } else
		goto fail;
	if (err < 0)
		goto fail;
	return 0;
fail:
	return -1;
}

int assembly_runtime_shutdown(void)
{
	int err = 0;
	if (!internals) {
		printd(DBG_WARNING, "assembly runtime not initialized\n");
		return 0;
	}
	if (internals->type == NODE_TYPE_MAIN)
		err = node_main_shutdown();
	else if (internals->type == NODE_TYPE_MINION)
		err = node_minion_shutdown();
	else if (internals->type == NODE_TYPE_MAPPER)
		err = node_mapper_shutdown();
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

/* pid = process requesting the assembly (the app, not runtime.c) */
asmid_t assembly_request(const struct assembly_hint *hint, pid_t pid)
{
	int err;
	struct assembly *assm = NULL;

	BUG(!internals);
	pthread_mutex_lock(&internals->lock);

	if (internals->type == NODE_TYPE_MAIN) {
		char hostname[HOST_LEN];
		gethostname(hostname, HOST_LEN);
		assm = __compose_assembly(hint, hostname);
		if (assm == NULL) {
			pthread_mutex_unlock(&internals->lock);
			goto fail;
		}
	}

	else if (internals->type == NODE_TYPE_MINION) {
		struct assembly *new_assm = calloc(1, sizeof(*new_assm));
		if (!new_assm) {
			goto fail;
		}
		// ask the main node for an assembly
		err = rpc_send_request(&internals->n.minion.rpc_conn, hint, new_assm);
		if (err < 0) { // could not acquire assembly
			free(new_assm);
			pthread_mutex_unlock(&internals->lock);
			goto fail;
		}
		assm = new_assm;
	}

    assm->pid = pid;

	list_add(&assm->link, &internals->assembly_list);
	pthread_mutex_unlock(&internals->lock);

	return assm->id;

fail:
	return INVALID_ASSEMBLY_ID;
}

int assembly_teardown(asmid_t id)
{
	int err, exit_errno;
	struct assembly *assm = NULL;

	BUG(!internals);

	pthread_mutex_lock(&internals->lock);
	assm = __find_assembly(id);
	if (!assm) {
		pthread_mutex_unlock(&internals->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	if (internals->type == NODE_TYPE_MINION) {
		// let the main node before we remove our state
		err = rpc_send_teardown(&internals->n.minion.rpc_conn, id);
		if (err < 0) {
			exit_errno = err; // failure could be so many things...
			pthread_mutex_unlock(&internals->lock);
			goto fail;
		}
	}
    else if (internals->type == NODE_TYPE_MAIN) {
        __decompose_assembly(assm);
    }

	list_del(&assm->link);
	//free(assm->cubins); // XXX this segvs
	assm->cubins = NULL;
	free(assm);

	pthread_mutex_unlock(&internals->lock);
	return 0;
fail:
	return exit_errno;
}

void assembly_print(asmid_t id)
{
	struct assembly *assm = NULL;
	int dev;

	BUG(!internals);

	pthread_mutex_lock(&internals->lock);
	assm = __find_assembly(id);
	if (!assm) {
		pthread_mutex_unlock(&internals->lock);
		return;
	}
	pthread_mutex_unlock(&internals->lock);

	printf("> Assembly --------\n");
	printf(">   ID      %04lu\n", assm->id);
	printf(">   PID     %04d\n", assm->pid);
	printf(">   vGPUs   %04d\n", assm->num_gpus);
	for (dev = 0; dev < assm->num_gpus; dev++) {
	printf(">       %02d maps to %d@%s\n",
				assm->mappings[dev].vgpu_id,
				assm->mappings[dev].pgpu_id,
				assm->mappings[dev].hostname);
	}
    printf(">   Hint\n");
    printf(">       vGPUs      %04d\n", assm->hint.num_gpus);
    if (assm->hint.mpi_group == 0)
        printf(">       MPI        n/a\n");
    else
        printf(">       MPI        %04d\n", assm->hint.mpi_group);
    printf(">       Policy     %s\n",
            assembly_hint_enum2policy(assm->hint.policy));
    printf(">       BatchSize  %04lu\n", assm->hint.batch_size);
    printf(">       NIC        %s\n",
            assembly_hint_enum2nic(assm->hint.nic_type));
}

// establish remote data paths, including vgpu ops
int assembly_map(asmid_t id)
{
	int err, exit_errno;
	struct assembly *assm = NULL;
	struct vgpu_mapping *vgpu = NULL;
	int vgpu_id;

	assm = assembly_find(id);
	if (!assm) {
		exit_errno = -EINVAL;
		goto fail;
	}

	for (vgpu_id = 0; vgpu_id < assm->num_gpus; vgpu_id++) {
		vgpu = &assm->mappings[vgpu_id];

		if (vgpu->fixation == VGPU_REMOTE) {
			vgpu->rpc = calloc(1, sizeof(*(vgpu->rpc)));
			if (!vgpu->rpc) {
				exit_errno = -ENOMEM;
				goto fail;
			}
			err = cuda_rpc_init(vgpu->rpc, assm->hint.batch_size);
			if (err < 0) {
				free(vgpu->rpc);
				exit_errno = -EIO;
				goto fail;
			}
			err = cuda_rpc_connect(vgpu->rpc, vgpu->ip, REMOTE_CUDA_PORT);
			if (err < 0) {
				free(vgpu->rpc);
				exit_errno = -EIO;
				goto fail;
			}
			vgpu->ops = rpc_ops;
			// Send our PID over, so the remote end can correctly
			// maintain CUBIN state.
			pid_t app_pid = getpid();
			err = conn_put(&vgpu->rpc->sockconn, &app_pid, sizeof(pid_t));
			if (err < 0) {
				cuda_rpc_close(vgpu->rpc);
				free(vgpu->rpc);
				exit_errno = -EIO;
				goto fail;
			}
		}

		else if (vgpu->fixation == VGPU_LOCAL) {
			vgpu->ops = exec_ops;
			vgpu->rpc = NULL;
		}
		
		else {
			BUG(1);
		}

	} // for each vgpu mapping

	assm->mapped = true;

	return 0;
fail:
	return exit_errno;
}

int asssembly_vgpu_is_local(asmid_t id, int vgpu_id, bool *answer)
{
    struct assembly *assm = NULL;
	struct vgpu_mapping *vgpu = NULL;

    if (!answer)
        return -1;

	if(!(assm = assembly_find(id)))
        return -1;

    if (vgpu_id >= assm->num_gpus)
        return -1;

    vgpu = &assm->mappings[vgpu_id];

    *answer = (vgpu->fixation == VGPU_LOCAL);
    return 0;
}

int assembly_read_hint(struct assembly_hint *hint)
{
    const char *val = getenv(HINT_ENV_VAR);
    if (!hint) {
        printd(DBG_ERROR, "NULL argument\n");
        return -1;
    }
    if (!val) {
        printf("> No hint file specified. Using default\n");
        memcpy(hint, &assembly_default_hint, sizeof(*hint));
        return 0;
    }
    return __do_read_hint(val, hint);
}

/*******************************************************************************
 * These functions (export/import) only exist because I (stupidly or not)
 * decided to make a stateful library out of this assembly code, and _also_
 * decided to make the backend runtime multi-process... so these functions are
 * needed to transfer state across address spaces.
 *
 * The idea of exporting and importing was a suggestion by Hrishikesh Amur.
 */

extern int export_assembly(assembly_key key, const struct assembly *assm);
int assembly_export(asmid_t id, assembly_key *key)
{
	int err, exit_errno;
	struct assembly *assm = NULL;

	BUG(!internals);

	pthread_mutex_lock(&internals->lock);
	assm = __find_assembly(id);
	if (!assm) {
		pthread_mutex_unlock(&internals->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	pthread_mutex_unlock(&internals->lock);

	if (assm->mapped) {
		exit_errno = -1;
		goto fail;
	}

	// TODO should I keep the lock while exporting?
    *key = getpid();
	err = export_assembly(*key, assm);
	if (err < 0) {
		exit_errno = -EIO;
		goto fail;
	}
	return 0;

fail:
	printd(DBG_ERROR, "failed\n");
	return exit_errno;
}

extern int import_assembly(assembly_key key, struct assembly *assm);
int assembly_import(asmid_t *id, assembly_key key)
{
	int err, exit_errno;
	struct assembly *assm = NULL;

	BUG(!internals);

	assm = calloc(1, sizeof(*assm));
	if (!assm) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	err = import_assembly(key, assm);
	if (err < 0) {
		if (err == -ENOENT)
			exit_errno = -EINVAL;
		else
			exit_errno = -EIO;
		goto fail;
	}

	// Unfortunately there are few more things that need to be "cleaned up"
	// before the state is consistent with what compose_assembly would return.
	// That includes anything that relies on pointers and heap-allocated data.
	INIT_LIST_HEAD(&assm->link);
	assm->cubins = malloc(sizeof(struct fatcubins));
	if (!assm->cubins) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	cubins_init(assm->cubins);

	BUG(assm->num_gpus < 1);

	pthread_mutex_lock(&internals->lock);
	list_add(&assm->link, &internals->assembly_list);
	pthread_mutex_unlock(&internals->lock);

	*id = assm->id;
	return 0;
fail:
	printd(DBG_ERROR, "Failed: %d\n", exit_errno);
	if (assm)
		free(assm);
	return exit_errno;
}

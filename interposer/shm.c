/**
 * @file shm.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 9, 2012
 * @brief Code used to connect with the localsink.
 */

// System includes
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Other project includes
#include <shmgrp.h>

// Project includes
#include <util/list.h>
#include <assembly.h>
#include <debug.h>

// Directory-immediate includes
#include "timing.h"

/*-------------------------------------- DEFINITIONS -------------------------*/

//! Amount of memory to allocate for each CUDA thread, in bytes.
//! XXX FIXME Put in checks to make sure the usercode isn't trying to copy more
//! than we allocate.
#define THREAD_SHM_SIZE					(512 << 20)

/**
 * Region of memory we've mapped in with the runtime. Each region is assigned to
 * a unique thread in the CUDA application.
 */
struct shm_region {
	struct list_head link;
	struct shmgrp_region shmgrp_region;
	pthread_t tid;	//! Application thread assigned this region
};

//! List of shm regions maintained with the assembly runtime.
struct shm_regions {
	struct list_head list;
	pthread_mutex_t lock; //! two-fold lock: this structure and the shmgrp API
};

//! Iterator loop for shm regions.
#define __shm_for_each_region(regions, region)	\
	list_for_each_entry(region, &((regions)->list), link)

//! Iterator loop for shm regions, allowing us to remove regions as we iterate.
#define __shm_for_each_region_safe(regions, region, tmp)	\
	list_for_each_entry_safe(region, tmp, &((regions)->list), link)

// All the __shm* functions should not be called directly. They should only be
// called by the other functions just below them, as any locking that needs to
// happen is handled correctly within those.

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct shm_regions *cuda_regions;

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

static inline struct shm_region *
__shm_get_region(struct shm_regions *regions, pthread_t tid)
{
	struct shm_region *region;
	__shm_for_each_region(regions, region)
		if (pthread_equal(region->tid, tid) != 0)
			return region;
	return NULL;
}

static inline void
__shm_add_region(struct shm_regions *regions, struct shm_region *region)
{
	list_add(&region->link, &regions->list);
}

static inline void
__shm_rm_region(struct shm_region *region)
{
	list_del(&region->link);
}

static inline bool
__shm_has_regions(struct shm_regions *regions)
{
	return !(list_empty(&regions->list));
}

/**
 * Create a new shared memory region with the runtime, and add a new entry to
 * the region list for us. This function must be thread-safe (newly detected
 * threads will ask for a new region, and will call this). If the tid has
 * already been allocated a memory region, we simply return that mapping.
 */
static void * __add_shm(size_t size, pthread_t tid)
{
	int err;
	struct shm_region *region;
	shmgrp_region_id id;
	region = __shm_get_region(cuda_regions, tid);
	if (region)
		return region->shmgrp_region.addr;
	region = calloc(1, sizeof(*region));
	if (!region) {
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	INIT_LIST_HEAD(&region->link);
	region->tid = tid;
	err = shmgrp_mkreg(ASSEMBLY_SHMGRP_KEY, size, &id);
	if (err < 0) {
		fprintf(stderr, "Error creating a new memory region with assembly\n");
		goto fail;
	}
	err = shmgrp_leader_region(ASSEMBLY_SHMGRP_KEY, id, &region->shmgrp_region);
	if (err < 0) {
		fprintf(stderr, "Error accessing existing shm region %d\n", id);
		goto fail;
	}
	__shm_add_region(cuda_regions, region);
	return region->shmgrp_region.addr;
fail:
	if (region)
		free(region);
	return NULL;
}

/*-------------------------------------- EXTERNAL FUNCTIONS ------------------*/

// Should be called within the first CUDA call interposed. It is okay to call
// this function more than once, it will only have an effect the first time.
int attach_assembly_runtime(void)
{
	int err;
	if (cuda_regions)
		return 0;
#ifdef TIMING
	timers_init();
	timers_start_attach();
#endif
	cuda_regions = calloc(1, sizeof(*cuda_regions));
	if (!cuda_regions) {
		fprintf(stderr, "Out of memory\n");
		return -1;
	}
	INIT_LIST_HEAD(&cuda_regions->list);
	err = shmgrp_init();
	if (err < 0) {
		fprintf(stderr, "Error initializing shmgrp state\n");
		return -1;
	}
	err = shmgrp_join(ASSEMBLY_SHMGRP_KEY);
	if (err < 0) {
		fprintf(stderr, "Error attaching to assembly runtime\n");
		return -1;
	}
#ifdef TIMING
	timers_stop_attach();
#endif
	// TODO Install a SIGINT handler so we can disconnect from the group/remove
	// any MQs we created.
	return 0;
}

// Should be called within the last CUDA call interposed. It is okay to call
// this function more than once, it will only have an effect the first time.
void detach_assembly_runtime(void)
{
	int err;
	struct shm_region *region, *tmp;
	if (!cuda_regions)
		return;
#ifdef TIMING
	timers_start_detach();
#endif
	if (__shm_has_regions(cuda_regions)) {
		__shm_for_each_region_safe(cuda_regions, region, tmp) {
			__shm_rm_region(region);
			err = shmgrp_rmreg(ASSEMBLY_SHMGRP_KEY, region->shmgrp_region.id);
			if (err < 0) {
				printd(DBG_ERROR, "Error destroying region %d\n",
						region->shmgrp_region.id);
			}
			free(region);
		}
	}
	free(cuda_regions);
	err = shmgrp_leave(ASSEMBLY_SHMGRP_KEY);
	if (err < 0)
		fprintf(stderr, "Error detaching from assembly runtime\n");
	err = shmgrp_tini();
	if (err < 0)
		fprintf(stderr, "Error uninitializing shmgrp state\n");
#ifdef TIMING
	timers_stop_detach();
#endif
}

//! Each interposed call will invoke this to locate the shared memory region
//! allocated to the thread calling it. If one does not exist, one is allocated.
//! Thus if there are no errors, this function will always return a valid
//! address.
void* get_region(pthread_t tid)
{
	void *addr;
	pthread_mutex_lock(&cuda_regions->lock);
	addr = __add_shm(THREAD_SHM_SIZE, tid);
	// XXX This memset seems to prevent certain race bugs with the localsink
	// threads.
	memset(addr, 0, sizeof(struct cuda_packet));
	pthread_mutex_unlock(&cuda_regions->lock);
	return addr;
}

/**
 * @file test_member.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-09
 */
#include <errno.h>
#include <shmgrp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRINTF(fmt, args...)			\
	do {								\
		printf("(pid %d): ", getpid());	\
		printf(fmt, ##args);			\
		fflush(stdout);					\
	} while(0)

#define FPRINTF(strm, fmt, args...)				\
	do {										\
		fprintf(strm, "(pid %d): ", getpid());	\
		fprintf(strm, fmt, ##args);				\
		fflush(strm);							\
	} while(0)

#define GRPKEY1	"cudarpc"

#define MAX_MEMBERS	50	//! Max number of simultaneous joins
#define MAX_REGIONS 20	//! Max number of regions to create per join

// TODO Add threading to create regions.

static int do_shm(int num_shm)
{
	int err;
	int shm_idx;
	shmgrp_region_id *ids = NULL;
	struct shmgrp_region *regions = NULL;

	ids = calloc(num_shm, sizeof(shmgrp_region_id));
	if (!ids)
		goto fail;
	regions = calloc(num_shm, sizeof(struct shmgrp_region));
	if (!regions)
		goto fail;

	for (shm_idx = 0; shm_idx < num_shm; shm_idx++) {

		PRINTF("new region with group %s\n", GRPKEY1);
		err = shmgrp_mkreg(GRPKEY1, 4096, &ids[shm_idx]);
		if (err < 0) {
			FPRINTF(stderr, "Error: shmgrp_mkreg(%s): %s\n",
					GRPKEY1, strerror(-err));
			goto fail;
		}

		PRINTF("Got region %d\n", ids[shm_idx]);
		err = shmgrp_leader_region(GRPKEY1, ids[shm_idx], &regions[shm_idx]);
		if (err < 0) {
			FPRINTF(stderr, "Error: shmgrp_leader_region: %s\n",
					strerror(-(err)));
			goto fail;
		}

		PRINTF("Region %d is mapped @ %p for %d bytes\n",
				regions[shm_idx].id, regions[shm_idx].addr,
				regions[shm_idx].size);

		usleep(250000); // Give leader some time to write into region.
		PRINTF("Leader wrote %lx\n", *((unsigned int*)regions[shm_idx].addr));

		PRINTF("Removing memory region %d from group %s\n",
				ids[shm_idx], GRPKEY1);
		err = shmgrp_rmreg(GRPKEY1, ids[shm_idx]);
		if (err < 0) {
			FPRINTF(stderr, "Error: shmgrp_rmreg(%s): %s\n",
					GRPKEY1, strerror(-err));
			goto fail;
		}
	}
	return 0;

fail:
	if (ids)
		free(ids);
	if (regions)
		free(regions);
	return -1;
}

static int do_join(bool perform_shm, int num_shm)
{
	int err;
	err = shmgrp_init();
	if (err < 0) {
		FPRINTF(stderr, "Error: shmgrp_init: %s\n", strerror(-err));
		return -1;
	}

	PRINTF("PID %d joining group %s\n", getpid(), GRPKEY1);
	err = shmgrp_join(GRPKEY1);
	if (err < 0) {
		FPRINTF(stderr, "Error: shmgrp_join(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}
	
	
	if (perform_shm) {
		err = do_shm(num_shm);
		if (err < 0) {
			FPRINTF(stderr, "Error do_shm: %d\n", err);
			return -1;
		}
	} else {
		// Prevent a race condition in the leader, joining/leaving too quickly.
		sleep(1);
	}

	PRINTF("Leaving group %s\n", GRPKEY1);
	err = shmgrp_leave(GRPKEY1);
	if (err < 0) {
		FPRINTF(stderr, "Error: shmgrp_leave(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	err = shmgrp_tini();
	if (err < 0) {
		FPRINTF(stderr, "Error: shmgrp_tini: %s\n", strerror(-err));
		return -1;
	}
	return 0;
}

// First argument is number of simultaneous member joins that should be
// performed. This is accomplished by forking n times to generate n children. If
// no arg is given, we assume no forking (i.e. one join).
int main(int argc, char *argv[])
{
	int err;
	int num_joins, num_shm;
	pid_t children[MAX_MEMBERS];
	int child;

	printf("\nThis test performs multi-process member testing with"
			" 1-N regions created and destroyed sequentially per process for a"
			" single leader (no intermixing of region creation/destruction,"
			" and no threading)\n\n");

	// Set value of num_joins
	if (argc == 3) {
		num_joins = atoi(argv[1]);
		num_shm = atoi(argv[2]);
		if (num_joins > MAX_MEMBERS || num_joins < 1) {
			fprintf(stderr, "Error: [num_joins]=%d is invalid. Must be 1-%d\n",
					num_joins, MAX_MEMBERS);
			return -1;
		}
		if (num_shm > MAX_REGIONS || num_shm < 0) {
			fprintf(stderr, "Error: [num_shm]=%d is invalid. Must be 0-%d\n",
					num_joins, MAX_REGIONS);
			return -1;
		}
	} else {
		printf("Usage: %s [num_joins num_shm]\n", argv[0]);
		printf("		where num_joins may be 1-%d\n", MAX_MEMBERS);
		printf("		where num_shm may be 0-%d\n", MAX_REGIONS);

		printf("\nExample: %s 5 3\n", argv[0]);
		printf("\twill fork 5 members, each creating then"
				" destroying one region 3 times\n\n");
		return -1;
	}

	PRINTF("parent\n");

	// Fork children to perform any joins and region creations
	for (child = 0; child < num_joins; child++) {
		pid_t pid = fork();
		if (pid == 0) { // child
			PRINTF("new child\n");
			// Pause a random amount of time before joining or leaving.
			srand(getpid());
			usleep(rand() % 5000000 + 250000);
			err = do_join((num_shm > 0), num_shm);
			if (err < 0)
				exit(-1);
			else
				exit(0);
		} else if (pid > 0) { // parent
			children[child] = pid;
		} else { // error
			FPRINTF(stderr, "Error forking: %s."
					" Kill any zombie processes you find.\n",
					strerror(errno));
			return -1;
		}
	}

	// Wait on all children one at a time.
	for (child = 0; child < num_joins; child++) {
		int status = 0;
		err = waitpid(children[child], &status, 0);
		if (err < 0) {
			FPRINTF(stderr, "Error invoking waitpid: %s\n",
					strerror(errno));
		}
		if (status != 0) {
			FPRINTF(stderr, "Error: child %d exited uncleanly\n",
					children[child]);
		} else {
			PRINTF("Child %d exited cleanly\n", children[child]);
		}
	}

	PRINTF("Test done.\n");
	return 0;
}

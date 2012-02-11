/**
 * @file test_leader.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-09
 * 
 * Create one group. Expect only one member to join, who will create one memory
 * region. Into this region we write a value. The member sleeps for 1 second to
 * give us time to do so. It will read this value from the region and print it
 * out.
 */
#include <shmgrp.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

// Locking for printf
pthread_mutex_t lock;
#define printf_locked(fmt, args...)		\
	do {								\
		pthread_mutex_lock(&lock);		\
		printf(fmt, ##args);			\
		pthread_mutex_unlock(&lock);	\
	} while(0)
#define fprintf_locked(err, fmt, args...)	\
	do {									\
		pthread_mutex_lock(&lock);			\
		fprintf(err, fmt, ##args);			\
		pthread_mutex_unlock(&lock);		\
	} while(0)

#define GRPKEY1	"cudarpc"

void cudarpc_shm_callback(shm_event e, pid_t pid, shmgrp_region_id id)
{
	struct shmgrp_region region;
	int err;
	memset(&region, 0, sizeof(region));

	switch (e) {
		case SHM_CREATE_REGION:
			err = shmgrp_member_region(GRPKEY1, pid, id, &region);
			if (err < 0) {
				fprintf_locked(stderr, "shm create: cannot find id %d\n", id);
			} else {
				printf_locked("shm create: pid %d made region %d @ %p for %d bytes\n",
						pid, id, region.addr, region.size);
				*((unsigned int *)region.addr) = 0xf005ba11;
			}
			break;
		case SHM_REMOVE_REGION:
			printf_locked("shm rem: pid %d closed region %d\n", pid, id);
			break;
	}
}

void cudarpc_callback(group_event e, pid_t pid)
{
	int err;

	if (e == MEMBERSHIP_JOIN) {
		err = shmgrp_establish_member(GRPKEY1, pid, cudarpc_shm_callback);
		if (err < 0) {
			fprintf_locked(stderr, "grp join: pid %d requested,"
					" but can't establish: %s\n", pid, strerror(-(err)));
		} else {
			printf_locked("grp join: member pid %d successful\n", pid);
		}
		// implement application-level setup here
	}
	else if (e == MEMBERSHIP_LEAVE) {
		err = shmgrp_destroy_member(GRPKEY1, pid);
		if (err < 0) {
			fprintf_locked(stderr, "grp leave: pid %d requested,"
					" but can't destroy: %s\n", pid, strerror(-(err)));
		}
		printf_locked("grp leave: member PID %d\n", pid);
		// implement application-level cleanup here
	}
}

int main(void)
{
	int err, sleeptime=30;

	pthread_mutex_init(&lock, NULL);

	err = shmgrp_init();
	if (err < 0) {
		fprintf_locked(stderr, "Error: shmgrp_init: %s\n", strerror(-err));
		return -1;
	}

	printf_locked("Opening group %s\n", GRPKEY1);
	err = shmgrp_open(GRPKEY1, cudarpc_callback);
	if (err < 0) {
		fprintf_locked(stderr, "Error: shmgrp_open(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	printf_locked("Sleeping for %d seconds."
			" You must execute the member now.\n", sleeptime);

	while(sleeptime-- > 0) {
		if (sleeptime %2 == 0) { // print every two seconds
			printf_locked("Sleeping... %d\n", sleeptime);
			fflush(stdout);
		}
		sleep(1);
	}

	printf_locked("Closing group %s\n", GRPKEY1);
	err = shmgrp_close(GRPKEY1);
	if (err < 0) {
		fprintf_locked(stderr, "Error: shmgrp_close(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	err = shmgrp_tini();
	if (err < 0) {
		fprintf_locked(stderr, "Error: shmgrp_tini: %s\n", strerror(-err));
		return -1;
	}
}

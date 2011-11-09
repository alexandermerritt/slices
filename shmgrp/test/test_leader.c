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

#define GRPKEY1	"cudarpc"
pid_t pid;
unsigned int value = 0xf005ba11;

void cudarpc_shm_callback(shm_event e, pid_t pid, shmgrp_region_id id)
{
	struct shmgrp_region region;
	int err;
	memset(&region, 0, sizeof(region));

	switch (e) {
		case SHM_CREATE_REGION:
			err = shmgrp_member_region(GRPKEY1, pid, id, &region);
			if (err < 0) {
				fprintf(stderr, "oopsie, PID %d created an shm %d,"
						" but we can't find it\n", pid, id);
			} else {
				printf("oh boy, PID %d wants to talk with us"
						" on region %d @ %p for %d bytes\n",
						pid, id, region.addr, region.size);
				printf("i'll tell it to play foosball\n");
				*((unsigned int *)region.addr) = value;
			}
			break;
		case SHM_REMOVE_REGION:
			printf("shucks, PID %d is closing region (%d)\n", pid, id);
			break;
	}
}

void cudarpc_callback(group_event e, pid_t pid)
{
	int err;

	if (e == MEMBERSHIP_JOIN) {
		err = shmgrp_establish_member(GRPKEY1, pid, cudarpc_shm_callback);
		if (err < 0)
			fprintf(stderr, "oops, PID %d can't be a member: %s\n",
					pid, strerror(-(err)));
		printf("hello hello to member PID %d\n", pid);
		// implement application-level setup here
	}
	else if (e == MEMBERSHIP_LEAVE) {
		printf("bye bye to member PID %d\n", pid);
		// implement application-level cleanup here
	}
}

int main(void)
{
	int err, sleeptime=30;

	err = shmgrp_init();
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_init: %s\n", strerror(-err));
		return -1;
	}

	printf("Opening group %s\n", GRPKEY1);
	err = shmgrp_open(GRPKEY1, cudarpc_callback);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_open(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	printf("Sleeping for %d seconds."
			" You must execute the member now.\n", sleeptime);

	while(sleeptime-- > 0) {
		if (sleeptime %2 == 0) // print every two seconds
			printf("Sleeping... %d\n", sleeptime), fflush(stdout);
		sleep(1);
	}

	printf("Closing group %s\n", GRPKEY1);
	err = shmgrp_close(GRPKEY1);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_close(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	err = shmgrp_tini();
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_tini: %s\n", strerror(-err));
		return -1;
	}
}

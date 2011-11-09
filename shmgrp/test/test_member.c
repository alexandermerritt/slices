/**
 * @file test_member.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-09
 */
#include <shmgrp.h>
#include <stdio.h>
#include <string.h>

#define GRPKEY1	"cudarpc"

int main(void)
{
	int err;
	shmgrp_region_id id;
	struct shmgrp_region region;

	err = shmgrp_init();
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_init: %s\n", strerror(-err));
		return -1;
	}

	printf("Joining group %s\n", GRPKEY1);
	err = shmgrp_join(GRPKEY1);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_join(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}
	
	printf("Creating new memory region with group %s\n", GRPKEY1);
	err = shmgrp_mkreg(GRPKEY1, 4096, &id);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_mkreg(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}
	printf("Got region %d, let's see where it is mapped\n", id);
	err = shmgrp_leader_region(GRPKEY1, id, &region);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_leader_region: %s\n",
				strerror(-(err)));
		return -1;
	}
	printf("Region %d is mapped @ %p for %d bytes\n",
			region.id, region.addr, region.size);

	printf("Sleeping 1 second for leader to give us a message...\n");
	sleep(1);

	printf("Leader tells us to play %lx\n", *((unsigned int*)region.addr));

	printf("Removing memory region %d from group %s\n",
			id, GRPKEY1);
	err = shmgrp_rmreg(GRPKEY1, id);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_rmreg(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	printf("Leaving group %s\n", GRPKEY1);
	err = shmgrp_leave(GRPKEY1);
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_leave(%s): %s\n",
				GRPKEY1, strerror(-err));
		return -1;
	}

	err = shmgrp_tini();
	if (err < 0) {
		fprintf(stderr, "Error: shmgrp_tini: %s\n", strerror(-err));
		return -1;
	}
}

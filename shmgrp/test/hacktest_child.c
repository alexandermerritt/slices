#include <shmgrp.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>

#define GRPKEY1	"cudarpc"

static void cudarpc_shm_callback(shm_event e, pid_t pid, shmgrp_region_id id)
{
	struct shmgrp_region region;
	int err;
	memset(&region, 0, sizeof(region));

	switch (e) {
		case SHM_CREATE_REGION:
			err = shmgrp_member_region(GRPKEY1, pid, id, &region);
			if (err < 0) {
				fprintf(stderr, "shm create: cannot find id %d\n", id);
			} else {
				printf("shm create: pid %d made region %d @ %p for %d bytes\n",
						pid, id, region.addr, region.size);
				*((unsigned int *)region.addr) = 0xb01dface;
			}
			break;
		case SHM_REMOVE_REGION:
			printf("shm rem: pid %d closed region %d\n", pid, id);
			break;
	}
}

// Declare prototypes of the special functions
extern int __shmgrp_insert_group(const char *key);
extern int __shmgrp_dump_group(const char *key);

static void sigterm_handler(int sig)
{
	; // Do nothing, just prevent it from killing us.
}

// Attach to a specific member joining the group created by hacktest_leader.
// argv[1] is the pid needed to establish a member. This binary can either be
// invoked directly via the command line, or fork/exec'd from hacktest_leader
// directly (the latter is already implemented).
int main(int argc, char *argv[])
{
	int err;
	struct sigaction action;
	sigset_t mask;

	if (argc != 2)
		return -1;

	pid_t pid = atoi(argv[1]);

	printf("Child attaching to pid %d\n", pid);
	fflush(stdout);

	// Block all signals
	sigfillset(&mask);
	sigprocmask(SIG_BLOCK, &mask, NULL);

	// Install handler for the termination signal.
	memset(&action, 0, sizeof(action));
	action.sa_handler = sigterm_handler;
	sigemptyset(&action.sa_mask);
	err = sigaction(SIGINT, &action, NULL);
	if (err < 0) {
		fprintf(stderr, "Error installing SIGINT handler\n");
		return -1;
	}

	shmgrp_init();

	// hackery to restore state in our library
	err = __shmgrp_insert_group(GRPKEY1);
	if (err < 0) {
		fprintf(stderr, "Error __inserting group\n");
		return -1;
	}

	err = shmgrp_establish_member(GRPKEY1, pid, cudarpc_shm_callback);
	if (err < 0) {
		fprintf(stderr, "grp join: pid %d requested,"
				" but can't establish: %s\n", pid, strerror(-(err)));
		return -1;
	}

	// Wait for SIGINT before we clean up.
	printf("Child waiting for term sig\n");
	sigfillset(&mask);
	sigdelset(&mask, SIGINT);
	sigsuspend(&mask);

	err = shmgrp_destroy_member(GRPKEY1, pid);
	if (err < 0) {
		fprintf(stderr, "grp leave: pid %d requested,"
				" but can't destroy: %s\n", pid, strerror(-(err)));
	}

	err = __shmgrp_dump_group(GRPKEY1);
	if (err < 0) {
		fprintf(stderr, "Error __dumping group: %s\n", strerror(-(err)));
		return -1;
	}

	printf("grp leave: member PID %d\n", pid);

	shmgrp_tini();
	return 0;
}

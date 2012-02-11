/**
 * @file hacktest_leader.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-15
 * 
 * Create one group. ASSUME A SINGLE MEMBER JOINING AT A TIME. For each member
 * that joins, fork/exec hacktest_child giving it the pid of the member joining.
 * hacktest_child will invoke the special functions implemented within the
 * bottom of leader.c to attach to the member joining.
 *
 * exec clears the state of the library across a fork, and restarts the
 * specified binary at main(). Thus shmgrp has no state when it loads, and the
 * special functions inject the necessary state needed to continue calling
 * establish_member.
 */
#include <shmgrp.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <signal.h>
#include <errno.h>

#define GRPKEY1	"cudarpc"

static void child_hackfunc(pid_t pid)
{
	int err;
	char pid_str[255];
	memset(pid_str, 0, sizeof(pid_str));
	snprintf(pid_str, sizeof(pid_str), "%d", pid);
	// Remember, the first argument is always the name of the program as it was
	// invoked.
	err = execl("hacktest_child", "./hacktest_child", pid_str, NULL);
	if (err < 0) {
		fprintf(stderr, "Error with execl: %s\n", strerror(errno));
	}
}

static void cudarpc_callback(group_event e, pid_t pid)
{
	int err;
	static pid_t child_pid;

	if (e == MEMBERSHIP_JOIN) {
		// give birth
		child_pid = fork();
		if (child_pid == 0) {
			child_hackfunc(pid);
			_exit(-1);
		} else if (child_pid < 0) {
			fprintf(stderr, "Error forking\n");
		}
	} else if (e == MEMBERSHIP_LEAVE) {
		int status;
		err = kill(child_pid, SIGINT);
		if (err < 0) {
			fprintf(stderr, "Error sending SIGINT to child %d\n", child_pid);
		}
		pid_t wait_pid = waitpid(child_pid, &status, 0);
		if (wait_pid != child_pid) {
			fprintf(stderr, "Error waiting on child\n");
		}
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

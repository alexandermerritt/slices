/**
 * @file backend.c
 * @brief The executor of cuda calls
 *
 * based on : void * nvback_rdomu_thread(void *arg)
 * from remote_gpu/nvidia_backend/remote/remote_api_wrapper.c
 *
 * @date Mar 8, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#include <pthread.h>
#include "connection.h"
#include <stdlib.h>			// malloc
#include "libciutils.h"		// mallocCheck
#include <stdio.h>			// fprintf
#include <string.h>
#include "debug.h"
#include "remote_packet.h"
#include "remote_api_wrapper.h"
#include <unistd.h>
#include <assert.h>

#include <common/libregistration.h>

/**
 * called when an interposed cuda process starts up and registers itself with us
 */
void notification_callback(enum callback_event e, regid_t id) {
	// libregistration should set up the shared memory areas
	// the regid can be used to obtain the shared memory areas
	printd(DBG_INFO, "INVOKED type=%u id=%u\n", e, id);
}

int main(){
	int err;

	// Initialize library <--> backend registration.
	err = reg_init(32); // # processes we expect to register
	if (err < 0) {
		fprintf(stderr, "Could not initialize library registration\n");
		return -1;
	}
	err = reg_callback(notification_callback);
	if (err < 0) {
		printd(DBG_ERROR, "Could not set registration callback\n");
	}

	// TODO
	// initialize a bunch of other things, such as worker threads, assembly, etc
	// etc
	sleep(15);

	err = reg_shutdown();
	if (err < 0) {
		fprintf(stderr, "Could not shutdown library registration\n");
		return -1;
	}

	return 0;
}

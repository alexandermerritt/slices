/**
 * @file testConnectionSender.c
 * @brief The sender part
 *
 * @date Sep 14, 2011
 * @author Magda Slawinska, aka Magic Magg, magg dot gatech at gmail dot edu
 */

#include "connection.h"
#include <stdio.h>
#include <stdlib.h>
#include "debug.h"

static void usage(char** argv){

	printf("Usage:\n"
			"\t %s prost.cc.gt.edu\n"
			"\t prost.cc.gt.edu is the host name you want to connect to\n\n"
			"You might need to setup the LD_LIBRARY_PATH e.g."
			"LD_LIBRARY_PATH=/nics/d/home/smagg/proj/wksp-ecl-cpp/2011-09-14-kidron-utils-rce/interposer/:/nics/d/home/smagg/opt/cunit212/lib/:/nics/d/home/smagg/opt/glib-2.28.7/lib interposer/testConnectionSender", argv[0]);
}

int
main(int argc, char *argv[]){

	if( 2 != argc ){
		usage(argv);
		exit(-1);
	}

	conn_t myconn;

	// at the beginning the connection is not valid
	myconn.valid = 0;
	char* hostname = argv[1];
	if( -1 == conn_connect(&myconn, hostname) ){
		printf("Test FAILED!\n");
		exit(-1);
	}
	myconn.valid = 1;
	char* message = "44";
	int mesg_size = 2+1;
	p_debug("Sending message: %s\n", message);
	if (1 != put(&myconn, message, mesg_size)) {
		p_debug( "Problems with sending the mesg.\n");
		printf("TEST FAILED!\n");
		return ERROR;
	}

	conn_close(&myconn);

	printf("TEST PASSED!\n");

	return 0;
}

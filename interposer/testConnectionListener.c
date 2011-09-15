/**
 * @file testConnectionListener.c
 *
 * @date Sep 14, 2011
 * @author Magda Slawinska, aka Magic Magg, magg dot gatech at gmail dot edu
 */

#include <stdio.h>
#include "connection.h"
#include <string.h>

int
main(int argc, char *argv[]){


	// connection for listening on a port
	// 2 conn_t cannot be allocated on stack (see comment on conn_t)
	// since cause a seg faults
	conn_t * pConnListen;
	conn_t * pConn;

	if( (pConnListen = conn_malloc(__FUNCTION__, NULL)) == NULL ) return NULL;
	// this allocates memory for the strm.rpkts, i.e., the array of packets
	if( (pConn = conn_malloc(__FUNCTION__, NULL)) == NULL ) return NULL;

	// set up the connection
	conn_localbind(pConnListen);
	conn_accept(pConnListen, pConn);   // blocking

	//recv the header describing the batch of remote requests
	char buf[3];
	if (1 != get(pConn, buf, sizeof(buf))) {
		return -1;
	}

	if( strcmp(buf, "44") == 0 )
		printf("Test PASSED: got %s\n", buf);
	else
		printf("Test FAILED\n");

	conn_close(pConn);
	conn_close(pConnListen);

	return 0;
}

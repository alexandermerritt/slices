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

/**
 * Allocates memory for the connection
 *
 * @param pFuncName h9olds the name of the function
 * @return the pointer to the connection, otherwise NULL
 */
conn_t * mallocConn(const char * const pFuncName){
	conn_t *pConn;

	pConn = malloc(sizeof(conn_t));
	if( mallocCheck(pConn, pFuncName, NULL) != 0){
		pthread_exit(NULL);
		pConn = NULL;
	}

	return pConn;
}


void * backend_thread(){
	// connection for listening on a port
	// 2 conn_t cannot be allocated on stack (see comment on conn_t)
	// since cause a seg faults
	conn_t * pConnListen;
	conn_t * pConn;

	int npkts = 0, npkts_q = 0; // total packets; pkt counter for moving data in/out of call buffer
	int retval = 0;

	// Network state
	// FIXME cuda_bridge_request_t == cuda_packet_t == rpkt_t, this is rediculous
	strm_hdr_t *hdr         = NULL;
	rpkt_t *rpkts           = NULL; // array of cuda packets
	rpkt_t *rpkt_in_ring    = NULL; // pointer within call buffer
	char *reqbuf            = NULL;
	char *rspbuf            = NULL;

	if( (pConnListen = mallocConn(__FUNCTION__ )) == NULL ) return NULL;
	if( (pConn = mallocConn(__FUNCTION__ )) == NULL ) return NULL;

	fprintf(stderr, "%s.%d: hey, here server thread!\n", __FUNCTION__, __LINE__);
	// set up the connection
	conn_localbind(pConnListen);
	conn_accept(pConnListen, pConn);   // blocking

	// Connection-related structures
	hdr = &pConn->strm.hdr;
	rpkts = pConn->strm.rpkts;
	reqbuf = pConn->request_data_buffer;
	rspbuf = pConn->response_data_buffer;

    while(1) {
    	/////////////////////////////
		//// RECEIVE THE REQUEST STREAM
		///////////////////////////////

		memset(hdr, 0, sizeof(strm_hdr_t));
		memset(rpkts, 0, MAX_REMOTE_BATCH_SIZE * sizeof(rpkt_t));
		pConn->request_data_size = 0;
		pConn->response_data_size = 0;

		//recv the header describing the batch of remote requests
		if (1 != get(pConn, hdr, sizeof(strm_hdr_t))) {
			break;
		}
		if (hdr->num_cuda_pkts <= 0) {
			printd(DBG_WARNING,
					"%s.%d: Received header specifying zero packets, ignoring\n", __FUNCTION__, __LINE__);
			continue;
		}

        //
        // Receive the entire batch.
        //
		npkts = hdr->num_cuda_pkts;
		printd(DBG_DEBUG, "%s.%d: received request header of size %lu. Expecting  %d packets.\n",
				__FUNCTION__, __LINE__, sizeof(rpkt_t), hdr->num_cuda_pkts);

		// let's assume that we have enough space. otherwise we have a problem
		printd(DBG_INFO, "%s, Expecting %d packets.\n", __FUNCTION__, npkts);
		if(1 != get(pConn, rpkts, hdr->num_cuda_pkts * sizeof(rpkt_t))) {
			break;
		}

		if(hdr->data_size > 0) {
		   printd(DBG_INFO, "%s.%d: Expecting a %d-byte payload.\n", __FUNCTION__, __LINE__, hdr->data_size);
		   if(1 != get(pConn, reqbuf, hdr->data_size)) {
		          break;
		   }
		   pConn->request_data_size = hdr->data_size;
		}

		nvbackGetDeviceCount_srv(&rpkts[0], pConn);
		printf("%s.%d: Hey backend: The getDeviceCount is: %ld and the method id is %d\n", __FUNCTION__, __LINE__, rpkts[0].args[0].argi,
				rpkts[0].method_id);
		//
		// Enqueue requests from request buffer to call buffer.
		// instead, we will try to call functions
		// now it's time to execute the request
		// so try to do what DOMU_SERVER does

		printd(DBG_INFO, "Moving network batch to call buffer\n");
		npkts_q = npkts;


		//
		// Dequeue responses from call buffer to response buffer.
		//

		// !!!!!!!!!!!!!!!!!!!!!!!!!!! whatever

		//
		// Send response back out.
		//

/*		if (!strm_expects_response(&pConn->strm)) {
			continue;
		} */


    }

//	pPacket = callocCudaPacket(__FUNCTION__, &error);

//	// get the packet
//	get(pConnSend, pPacket, sizeof(pPacket));



//	// execute the remote call
//	if ( __nvback_cudaRegisterFatBinary(pPacket) != CUDA_SUCCESS ){
//		fprintf(stderr, "%s, Server talking: problems with executing: __nvback_cudaRegisterFatBinary",
//				__FUNCTION__);
//	}
	// nvback_rdomu_thread(void *arg) should be executor

	conn_close(pConnListen);
	conn_close(pConn);

	free(pConnListen);
	free(pConn);

	return NULL;
}

int main(){
	// thread that listens for the incoming connections
	pthread_t thread;

	pthread_create(&thread, NULL, &backend_thread, NULL);
	pthread_join(thread, NULL);

	printf("server thread says you bye!\n");
	return 0;
}

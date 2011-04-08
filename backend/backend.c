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


void * backend_thread(){
	// connection for listening on a port
	// 2 conn_t cannot be allocated on stack (see comment on conn_t)
	// since cause a seg faults
	conn_t * pConnListen;
	conn_t * pConn;

	//int retval = 0;

	// Network state
	// FIXME cuda_bridge_request_t == cuda_packet_t == rpkt_t, this is rediculous
	strm_hdr_t *hdr         = NULL;
	rpkt_t *rpkts           = NULL; // array of cuda packets
//	char *reqbuf            = NULL;
//	char *rspbuf            = NULL;


	if( (pConnListen = conn_malloc(__FUNCTION__, NULL)) == NULL ) return NULL;
	// this allocates memory for the strm.rpkts, i.e., the array of packets
	if( (pConn = conn_malloc(__FUNCTION__, NULL)) == NULL ) return NULL;


	fprintf(stderr, "**************************************\n");
	fprintf(stderr, "%s.%d: hey, here server thread!\n", __FUNCTION__, __LINE__);
	fprintf(stderr, "**************************************\n");

	// set up the connection
	conn_localbind(pConnListen);
	conn_accept(pConnListen, pConn);   // blocking

	// Connection-related structures
	hdr = &pConn->strm.hdr;
	rpkts = pConn->strm.rpkts;   // an array of packets
	// these are buffers for extra data transferred as
	// a request or response (in addition to cuda_packet_t
	// e.g. extra data are returned to cudaGetDeviceProperties
	// i.e., the structure of the
//	reqbuf = pConn->request_data_buffer;  // an array of characters
//	rspbuf = pConn->response_data_buffer; // an array of characters

    while(1) {

    	printd(DBG_INFO, "------------------New RPC--------------------\n");

		memset(hdr, 0, sizeof(strm_hdr_t));
		memset(rpkts, 0, MAX_REMOTE_BATCH_SIZE * sizeof(rpkt_t));
		pConn->request_data_size = 0;
		pConn->response_data_size = 0;

		//recv the header describing the batch of remote requests
		if (1 != get(pConn, hdr, sizeof(strm_hdr_t))) {
			break;
		}
		printd(DBG_DEBUG, "%s: received request header. Expecting  %d packets. And extra request buffer of data size %d\n",
						__FUNCTION__,  hdr->num_cuda_pkts, hdr->data_size);

		if (hdr->num_cuda_pkts <= 0) {
			printd(DBG_WARNING,
					"%s: Received header specifying zero packets, ignoring\n", __FUNCTION__);
			continue;
		}

        // Receive the entire batch.
		// first the the packets, then the extra data (reqbuf)
		//
		// let's assume that we have enough space. otherwise we have a problem
		// pConn allocates the buffers for incoming cuda packets
		// so we should be fine
		if(1 != get(pConn, rpkts, hdr->num_cuda_pkts * sizeof(rpkt_t))) {
			break;
		}
		printd(DBG_INFO, "%s: Received %d packets, each of size of (%lu) bytes\n",
				__FUNCTION__, hdr->num_cuda_pkts, sizeof(rpkt_t));

		printd(DBG_INFO, "%s: Received method %s (method_id %d) from Thr_id: %lu.\n",
				__FUNCTION__, methodIdToString(rpkts[0].method_id),
				rpkts[0].method_id, rpkts[0].thr_id);


		// receiving the request buffer if any
		if(hdr->data_size > 0){

			printd(DBG_INFO, "%s: Expecting data size/Buffer: %u, %d.\n", __FUNCTION__,
											 hdr->data_size, pConn->request_data_size);
			// let's assume that the expected amount of data will fit into
			// the buffer we have (size of pConn->request_data_buffer
			assert(hdr->data_size <= TOTAL_XFER_MAX);
			if(1 != get(pConn, pConn->request_data_buffer, hdr->data_size)){
				break;
			}
			pConn->request_data_size = hdr->data_size;
			printd(DBG_INFO, "%s. %d: Received request buffer (%d bytes)\n",
					__FUNCTION__, __LINE__, pConn->request_data_size);
		}

		// execute the request
		pkt_execute(&rpkts[0], pConn);

		// we need to send the one response packet + response_buffer if any

		// @todo you need to check if he method needs to send
		// and extended response - you need to provide the right
		// data_size

		if( strm_expects_response(&pConn->strm) ){

			// send the header about response
			printd(DBG_DEBUG, "pConn->response_data_size %d\n", pConn->response_data_size);
			if (conn_sendCudaPktHdr(&*pConn, 1, pConn->response_data_size) == ERROR) {
				printd(DBG_INFO, "%s.%d: Error after : Sending the CUDA packet response header: Quitting ... \n",
						__FUNCTION__, __LINE__);
				break;
			}

			// send the response as a simple cuda packet
			if (1 != put(pConn, rpkts, sizeof(rpkt_t))) {
				printd(DBG_INFO, "%s.%d: Error after : Sending CUDA response packet: Quitting ... \n",
						__FUNCTION__, __LINE__);
				break;
			}
			printd(DBG_INFO, "%s.%d: Response Packet sent.\n", __FUNCTION__, __LINE__);

			// send the data if have anything to send
			if( pConn->response_data_size > 0 ){
				if (1 != put(pConn, pConn->response_data_buffer, pConn->response_data_size)) {
					printd(DBG_INFO, "%s.%d: Error after : Sending accompanying response data: Quitting ... \n",
							__FUNCTION__, __LINE__);
					break;
				}
				printd(DBG_INFO, "%s.%d: Response buffer sent (%d) bytes.\n",
						__FUNCTION__, __LINE__, pConn->response_data_size);
			}
		}
    }

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

	printf("server thread says you bye bye!\n");
	return 0;
}

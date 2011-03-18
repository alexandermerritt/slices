/**
 * @file remote_api_wrapper.c
 * @brief Constants for the network module, copied from
 * remote_gpu/network/remote_api_wrapper.c and
 * edited by me MS to make it more independent
 *
 * @date Feb 25, 2011
 * @author Magda Slawinska, magg@gatech.edu
 */

/*#ifndef _GNU_SOURCE
#	define _GNU_SOURCE 1
#endif */

#include "remote_api_wrapper.h"
#include "debug.h"
#include <cuda.h>			// for CUDA_SUCCESS
#include <cuda_runtime_api.h> // for cuda calls, e.g., cudaGetDeviceCount, etc
#include "libciutils.h"
#include "method_id.h"
#include "remote_api_wrapper.h"

#include "remote_packet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>
#include "connection.h"


#include "local_api_wrapper.h"


//! Right now the host where we are connecting to (where clients, ie, *_rpc connects
//! to
#ifndef REMOTE_HOSTNAME
#	define REMOTE_HOSTNAME "cuda2.cc.gt.atl.ga.us"
#endif

//! open the connection forever; if you open and then close the
//! connection the program stops to work, so you have to make it static
static conn_t myconn;

///////////////////
// RPC CALL UTILS//
///////////////////

/**
 * @brief Process an entire batched set of CUDA API calls ("stream") to the remote host.
 *
 * We add the specific packet to the buffer that is maintained by the connection.
 * We also check if there are some requests in the reqbuf and we add this to our
 * connection. Then we send the connection data. TODO here I have problems, since
 * conn_t is a connection, and not the data so it should be called something
 * else (sending_endpoint (?) ).
 * It also deals with response. Since the connection can expect responses. Then
 * the response is passed through packet and rspbuf.
 *
 * @param myconn (out) connection will provide where the
 * @param packet (inout) the packet we will add to our buffer stream defined in
 *        myconn; 'in parameter' if we are sending the packet
 *        'out parameter' if we are getting a response
 * @param reqbuf (in) from this buffer data are added to myconn
 * @param reqbuf_size (in) size of the request buffer
 * @param rspbuf (out) we copy here a response we got
 * @param rspbuf_size (out) the size of the response buffer
 */
int do_cuda_rpc(cuda_packet_t *packet,
		void *reqbuf,
		int reqbuf_size,
		void *rspbuf,
		int rspbuf_size) {

/*	int retval = OK;
	strm_hdr_t *pHdr;  // declared to help with my connection
	rpkt_t * pRpkts;   // declared to help with connections

	// apparently the myconn has something interesting already, since
	// it previously was the domu_info or something, now we create
	// a connection here so, we have a problem since it is empty
	//conn_t * myconn;

	//if( (myconn = conn_malloc( __FUNCTION__, "connections")) == NULL ){
	//	exit(-1);
	//}
	conn_connect(myconn, REMOTE_HOSTNAME);

	pHdr = myconn.strm.hdr;
	// right now we are sending only one packet, normally it should
	// contain a current number of packets
	pHdr->num_cuda_pkts = 0;
	pRpkts = myconn.strm.rpkts;

	// send the header with info of what is the size
	// 1 is the normal put exit
	if (1 != put(myconn, myconn.strm.hdr, sizeof(strm_hdr_t))) {
		retval = ERROR;
		goto exit_do_cuda_rpc; // @todo change it
	}

	// I guess pHdr contains the number in the stream but right now
	// we only send one packet so increasing it doesn't matter
	memcpy(&pRpkts[pHdr->num_cuda_pkts], packet, sizeof(cuda_packet_t));
	pHdr->num_cuda_pkts++;
	pHdr->data_size = sizeof(cuda_packet_t);

	printd(DBG_DEBUG, "%s.%d: Send request header for %d packets.\n", __FUNCTION__, __LINE__, pHdr->num_cuda_pkts);

	//send request header (including data size) for the batch of remote packets
	if (1 != put(myconn, pHdr, sizeof(strm_hdr_t))) {
		retval = ERROR;
		goto exit_do_cuda_rpc; // @todo change it
	}

	printd(DBG_DEBUG, "%s.%d: Send request data %d bytes. \n", __FUNCTION__, __LINE__, pHdr->data_size);
	// send the request data
	if (1 != put(myconn, (char *) pRpkts, pHdr->data_size)) {
		retval = ERROR;
		goto finalize_do_cuda_rpc;
	}
	printd(DBG_DEBUG, "%s.%d: Here. \n", __FUNCTION__, __LINE__);
*/
	// -------------------------------

/*	// Add to stream packet. TODO use the appropriate function
	// add packet to the stream
	hdr->num_cuda_pkts++;
	memcpy(&rpkts[hdr->num_cuda_pkts - 1], packet, sizeof(cuda_packet_t));
	// set the data size
	hdr->data_size=sizeof(*packet);

	// @todo right now forget about streaming
	if (req_strm_has_data(&myconn->strm)) {
		assert(reqbuf && reqbuf_size);
		memcpy(myconn->request_data_buffer + hdr->data_size, reqbuf,
				reqbuf_size);
		rpkts[hdr->num_cuda_pkts - 1].ret_ex_val.data_unit = hdr->data_size;
		printd(DBG_DEBUG, "pkts[%d].ret_ex_val.data_unit = %u\n", hdr->num_cuda_pkts-1,
				rpkts[hdr->num_cuda_pkts-1].ret_ex_val.data_unit);
		hdr->data_size += reqbuf_size;
	} else {
		rpkts[hdr->num_cuda_pkts - 1].ret_ex_val.data_unit = -1;
	}
	if (!strm_flush_needed(&myconn->strm)) {
		retval = 0;
		packet->ret_ex_val.err = 0;
		goto exit_do_cuda_rpc;
		// @todo change it
	}
*/
	/////////////////////////////
	// SEND THE REQUEST STREAM
	/////////////////////////////

	/*
	 // ASSUMPTION: request data and data size are properly populated before calling this enqueue
	 // function
	 if(req_strm_has_data(&myconn->strm))
	 {
	 assert(reqbuf && reqbuf_size);
	 hdr->data_size = reqbuf_size;
	 }
	 */
/*	printd(DBG_DEBUG, "%s.%d: Send request header for %d packets.\n", __FUNCTION__, __LINE__, hdr->num_cuda_pkts);

	//send request header (including data size) for the batch of remote packets
	if (1 != put(myconn, hdr, sizeof(strm_hdr_t))) {
		retval = -1;
		goto finalize_do_cuda_rpc;		// @todo change it
	}

	// ASSUMPTION: request data and data size are properly populated before calling this enqueue
	// function
	//if(req_strm_has_data(&myconn->strm)) {
	if (hdr->data_size > 0) {
		//assert(reqbuf && reqbuf_size);
		printd(DBG_DEBUG, "Send request data %d bytes. \n", hdr->data_size);
		// send the request data
		if (1 != put(myconn, (char *) myconn->request_data_buffer, hdr->data_size)) {
			retval = -1;
			goto finalize_do_cuda_rpc;
		}
*/
		/*
		 printd(DBG_DEBUG, "REQUEST DATA SENT (%d bytes):\n", hdr->data_size);
		 for(i = 0; i < hdr->data_size; i++)
		 printd(DBG_DEBUG, "%c", myconn->request_data_buffer[i]);
		 printd(DBG_DEBUG,"\n");
		 */
/*	}
	printd(DBG_DEBUG, "%s.%d: Here.\n", __FUNCTION__, __LINE__);

	if (!strm_expects_response(&myconn->strm)) {
		retval = 0;
		packet->ret_ex_val.err = 0;
		goto finalize_do_cuda_rpc;
	}

	memset(hdr, 0, sizeof(strm_hdr_t));
	memset(rpkts, 0, MAX_REMOTE_BATCH_SIZE * sizeof(rpkt_t));

    /////////////////////////////
    // RECEIVE THE RESPONSE STREAM
    /////////////////////////////

    //recv response header for the batch
    if(1 != get(myconn, hdr, sizeof(strm_hdr_t))) {
        retval = -1;
        goto finalize_do_cuda_rpc;
    }

    printd(DBG_DEBUG, "received response header. Expecting %d packets in response batch\n",
            hdr->num_cuda_pkts);

    if(hdr->data_size != (unsigned int) rspbuf_size) {
        // NEED NOT BE AN ERROR
        printd(DBG_WARNING, "response size EXPECTED = %d. to be received=%d. \n", rspbuf_size,
                hdr->data_size);
    }

    assert(hdr->num_cuda_pkts == 1);

    printd(DBG_INFO, "received response batch. %d packets.\n", hdr->num_cuda_pkts);

    //copy back the received response packet to given request packet
    // assert(1 == hdr->num_cuda_pkts);
    memcpy(packet, &rpkts[0], sizeof(cuda_packet_t));

    //assert(0 == rpkts[hdr->num_cuda_pkts -1].ret_ex_val.data_unit);

    if(packet->method_id == __CUDA_REGISTER_FAT_BINARY) {
        printd(DBG_DEBUG, "FAT CUBIN HANDLE: registered %p.\n", packet->ret_ex_val.handle);
    }

    if (rsp_strm_has_data(&myconn->strm)) {
		assert(rspbuf && rspbuf_size);
		// recv the response data
		if (1 != get(myconn, rspbuf, hdr->data_size)) {
			retval = -1;
			goto finalize_do_cuda_rpc;
		}

        printd(DBG_INFO, "received response data. %d bytes. \n", hdr->data_size);
    }
*/
/*finalize_do_cuda_rpc:

	    memset(pHdr, 0, sizeof(strm_hdr_t));
	    memset(pRpkts, 0, MAX_REMOTE_BATCH_SIZE*sizeof(rpkt_t));

exit_do_cuda_rpc:
	conn_close(myconn);
	free(myconn);
	return retval;
	*/
	return 0;
}

/**
 * closes the connection and frees the memory occupied by the connection
 * @param pConn (inout) The connection to be closed; changed to NULL
 * @param exit_code (in) What will be returned
 * @return exit_code indicates if we want this function to indicate the erroneous
 *        behaviour or not @see do_cuda_rpc1
 *
 * @todo check with passing pointers to functions
 */
int l_cleanUpConn(conn_t * pConn, const int exit_code){
//	conn_close(pConn);
//	free(pConn);
//	pConn = NULL;

	return exit_code;
}
/**
 * executes the cuda call over the network
 * This is the entire protocol of sending and receiving requests.
 * Including data send as arguments, as well as extra responses
 * @param pPacket the packet that contains data to be send and executed over the
 * network
 * @param reqbuf (in) from this buffer data are added to myconn
 * @param reqbuf_size (in) size of the request buffer
 * @param rspbuf (out) we copy here a response we got from the server
 * @param rspbuf_size (out) the size of the response buffer
 *
 * @return OK everything went OK,
 *         ERROR if something went wrong
 */
int do_cuda_rpc1(cuda_packet_t *packet, void * reqbuf, const int reqbuf_size,
		void * rspbuf, const int rspbuf_size) {

	strm_hdr_t *pHdr;  // declared to help with my connection, will be a pointer
					   // to the header in my packet I want to send
	rpkt_t * pRpkts;   // declared to help with connections
	size_t rpkt_size = sizeof(rpkt_t);


	printd(DBG_DEBUG, "reqbuf %p, reqbuf_size %d, rspbuf %p, rspbuf_size %d\n",
			reqbuf, reqbuf_size, rspbuf, rspbuf_size);

	// connect if not connected, otherwise reuse the connection
	// if you close and open the connection, the program exits; a kind of
	// singleton
	if( 0 == myconn.valid){
		conn_connect(&myconn, REMOTE_HOSTNAME);
		myconn.valid = 1;
	}

	// for simplicity we use aliases for particular fields of the myconn
	pHdr = &myconn.strm.hdr;
	pRpkts = myconn.strm.rpkts;

	//pRpkts[0].ret_ex_val.data_unit = sizeof(rpkt_t);
	pHdr->num_cuda_pkts = 1;

	//if( req_strm_has_data(&myconn.strm) ){
	if( reqbuf_size > 0) {
		assert(reqbuf && reqbuf_size);
		assert(reqbuf_size <= TOTAL_XFER_MAX);
		memcpy(myconn.request_data_buffer, reqbuf, reqbuf_size);
		printd(DBG_DEBUG, "pkts[0].ret_ex_val.data_unit = %u\n",
				pRpkts[0].ret_ex_val.data_unit);
		pHdr->data_size = reqbuf_size;

		// @todo I changed the order in this section compared to the original
		// version; actually I do not understand why it is done and
		// I guess it indicates that the packet will come with extra request
		// buffer
		// store the size of the request buffer
		pRpkts[0].ret_ex_val.data_unit = pHdr->data_size;
	} else {
		// i guess it means that we will not send an request buffer
		pRpkts[0].ret_ex_val.data_unit = -1;
	}

	if ( conn_sendCudaPktHdr(&myconn, 1, reqbuf_size) == ERROR ){
		return l_cleanUpConn(&myconn, ERROR);
	}

	// now we are preparing for sending a cuda packet
	// start with  copying the packet into a contiguous space
	memcpy(&pRpkts[0], packet, rpkt_size);

	// send the packet
	if (1 != put(&myconn, (char *) pRpkts, pHdr->num_cuda_pkts * rpkt_size)) {
		return l_cleanUpConn(&myconn, ERROR);
	}

	// now send the extra request buffer if any
	if( pHdr->data_size > 0 ){
		assert(reqbuf && reqbuf_size);
		if(1 != put(&myconn, myconn.request_data_buffer, pHdr->data_size)) {
			return l_cleanUpConn(&myconn, ERROR);
		}
		printd(DBG_DEBUG, "Request buffer sent (%d bytes).\n", pHdr->data_size);
	}

	// now check if we have some extra response data, i.e.
	// check if the packet expects the response, i.e.,
	// the decision if we expect the response is hard-coded
	// in strm_expects_response
	if (!strm_expects_response(&myconn.strm)) {
		// apparently this indicates that the strm doesn't expect the response
		//packet->ret_ex_val.err = 0; //@todo it seems to be unnecessary
		return l_cleanUpConn(&myconn, OK);
	}

	// so we are expecting the response; ok now get the response
	memset(pHdr, 0, sizeof(strm_hdr_t));
	memset(pRpkts, 0, MAX_REMOTE_BATCH_SIZE * rpkt_size);

	// recv response header for the batch
	// the header will contain the size of the extra response
	// data if any
	if (1 != get(&myconn, pHdr, sizeof(strm_hdr_t))) {
		return l_cleanUpConn(&myconn, ERROR);
	}

	printd(DBG_DEBUG, "%s.%d: received response header. Expecting %d packets and extra response size of %u in response batch\n",
			__FUNCTION__, __LINE__, pHdr->num_cuda_pkts, pHdr->data_size);

	assert(pHdr->num_cuda_pkts == 1);

	// recv the  batch of the responses
	if(1 != get(&myconn, pRpkts, pHdr->num_cuda_pkts * rpkt_size)) {
		return l_cleanUpConn(&myconn, ERROR);
	 }

	printd(DBG_DEBUG, "%s.%d: Received response batch. %d packets\n",
			__FUNCTION__, __LINE__, pHdr->num_cuda_pkts);

	//copy back the received response packet to given request packet
	memcpy(packet, &pRpkts[0], rpkt_size);

	if(packet->method_id == __CUDA_REGISTER_FAT_BINARY) {
		printd(DBG_DEBUG, "%s.%d: FAT CUBIN HANDLE: registered %p.\n",
				__FUNCTION__, __LINE__, packet->ret_ex_val.handle);
	}

	// check if we need to receive an extra buffer of response
	//if( rsp_strm_has_data(&myconn.strm) ){
	if( pHdr->data_size > 0 ){
		assert(rspbuf && rspbuf_size);
		// actually pHdr->data_size should be equal to rspbuf_size
		assert((int) pHdr->data_size <= rspbuf_size );

		// ok, get the data, data will be stored in the rsp_buf
		// first receive the header
		if(1 != get(&myconn, rspbuf, (int) pHdr->data_size)) {
			return l_cleanUpConn(&myconn, ERROR);
		}
	}

	// finalize do_cuda_rpc
	memset(pHdr, 0, sizeof(strm_hdr_t));
	memset(pRpkts, 0, MAX_REMOTE_BATCH_SIZE*rpkt_size);

	return l_cleanUpConn(&myconn, OK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// REMOTE FUNCTIONS (CLIENT)
//
// They should convert dom_info + cpkt -> remote packet stream
//
////////////////////////////////////////////////////////////////////////////////////////////////////
int nvbackCudaGetDeviceCount_rpc(cuda_packet_t *packet){
    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // originally it was called the below line, since I have trouble
    // with running this I do this right now step by step
    //do_cuda_rpc(packet, NULL, 0, NULL, 0);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    do_cuda_rpc1(packet, NULL, 0, NULL, 0);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaGetDeviceProperties_rpc(cuda_packet_t *packet){
    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    // we need to provide a buffer where we will copy the data from
    // cudaDeviceProp structure we will get from the server
    // actually this is in pPacket->args[0].argp
    // packet->args[2].argi-> the size of the buffer required
    // we will interpret the pPacket->args[0].argp as argui (see explanation
    // in cudaGetDeviceProperies in libci.c why
    printd(DBG_DEBUG, "Response Buffer: pointer %p, size %ld\n", (void*)packet->args[0].argui, packet->args[2].argi);
    do_cuda_rpc1(packet, NULL, 0, (void *) packet->args[0].argui, packet->args[2].argi);

    return (packet->ret_ex_val.err == 0) ? OK : ERROR;
}

int nvbackCudaFree_rpc(cuda_packet_t *packet){

    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    do_cuda_rpc1(packet, NULL, 0, NULL, 0);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaMalloc_rpc(cuda_packet_t *packet){

    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);
    // clear the packet, we are also sending the size of
    // the memory to allocate
    packet->args[0].argp = NULL;
    do_cuda_rpc1(packet, NULL, 0, NULL, 0);

    printd(DBG_DEBUG,"%s: devPtr is %p",__FUNCTION__, packet->args[0].argp);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaSetupArgument_rpc(cuda_packet_t *packet){
	printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
	            packet->ret_ex_val.err, packet->method_id);

	do_cuda_rpc1(packet, (void *)packet->args[0].argp, packet->args[1].argi, NULL, 0);

	return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaConfigureCall_rpc(cuda_packet_t *packet){
    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    do_cuda_rpc1(packet, NULL, 0, NULL, 0);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int __nvback_cudaRegisterFatBinary_rpc(cuda_packet_t *packet) {
	// @todo currently not called
	printd(DBG_DEBUG, "%s: CUDA_ERROR=%d before RPC on method %d\n", __FUNCTION__,
			packet->ret_ex_val.err, packet->method_id);

	do_cuda_rpc(packet, (void *) packet->args[0].argui, packet->args[1].argi,
			NULL, 0);

	return CUDA_SUCCESS;
}



/////////////////////////
// SERVER SIDE CODE
/////////////////////////

int nvbackCudaGetDeviceCount_srv(cuda_packet_t *packet, conn_t * pConn){
    int devCount = 0;

    // just call the function
    packet->ret_ex_val.err = cudaGetDeviceCount(&devCount);
    packet->args[0].argi = devCount;

    printd(DBG_DEBUG, "%s.%d: CUDA_ERROR=%p for method id=%d after calling method\n",
    		__FUNCTION__, __LINE__, packet->ret_ex_val.handle, packet->method_id);
    return OK;
}

int nvbackCudaGetDeviceProperties_srv(cuda_packet_t *packet, conn_t * pConn){
	struct cudaDeviceProp * prop = (struct cudaDeviceProp *)pConn->response_data_buffer;

    pConn->response_data_size = sizeof(struct cudaDeviceProp);

    packet->ret_ex_val.err = cudaGetDeviceProperties(prop, packet->args[1].argi);

    // I guess you need to pack the change somehow the do_cuda_rpc1
    // to use and send the response_data_buffer
    printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaMalloc_srv(cuda_packet_t * packet, conn_t * pConn){
	// just call the function
	packet->args[0].argp = NULL;

	packet->ret_ex_val.err = cudaMalloc(&packet->args[0].argp, packet->args[1].argi);

    printd(DBG_DEBUG,"%s: devPtr is %p, size %ld\n",__FUNCTION__,packet->args[0].argp, packet->args[1].argi);
    printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d after execution\n",
    		packet->ret_ex_val.handle, packet->method_id);
	return OK;
}

int nvbackCudaFree_srv(cuda_packet_t *packet, conn_t *pConn){
	printd(DBG_DEBUG,"%s: devPtr is %p\n",__FUNCTION__,packet->args[0].argp);
    packet->ret_ex_val.err = cudaFree(packet->args[0].argp);
    printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaSetupArgument_srv(cuda_packet_t *packet, conn_t *pConn){
	void *arg = (void*) ((char *)pConn->request_data_buffer + packet->ret_ex_val.data_unit);
	packet->ret_ex_val.err = cudaSetupArgument( arg,
	            packet->args[1].argi,
	            packet->args[2].argi);
    printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaConfigureCall_srv(cuda_packet_t *packet, conn_t *pConn){
	packet->ret_ex_val.err = cudaConfigureCall( packet->args[0].arg_dim,
            packet->args[1].arg_dim,
            packet->args[2].argi,
            (cudaStream_t) packet->args[3].argi);

    printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int __nvback_cudaRegisterFatBinary_srv(cuda_packet_t *packet, conn_t * myconn){

	__cudaFatCudaBinary * pFatC;
	void ** pFatCHandle;

    pFatC = copyFatBinary((__cudaFatCudaBinary *)((char *)myconn->request_data_buffer + packet->ret_ex_val.data_unit));

	// call __cudaRegisterFatBinary; otherwise the compiler complaints that
	// undefined reference to __cudaRegisterFatBinary
    pFatCHandle = __cudaRegisterFatBinary(pFatC);

    packet->args[1].argp = pFatC;
    packet->ret_ex_val.handle = pFatCHandle;

    printd(DBG_DEBUG, "%s: FATCUBIN HANDLE: registered %p\n", __FUNCTION__, pFatCHandle);
    return CUDA_SUCCESS;
}

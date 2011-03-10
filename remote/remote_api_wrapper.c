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
	conn_t * myconn = malloc(sizeof(conn_t));
	int retval = OK;
	// @todo change it
	if (mallocCheck(myconn, __FUNCTION__, "Can't allocate memory for the connection. Quitting ...") != 0) {
		exit(-1);
	}
	conn_connect(myconn, REMOTE_HOSTNAME);

	strm_hdr_t *hdr;
	rpkt_t *rpkts;

	hdr = &myconn->strm.hdr;
	rpkts = myconn->strm.rpkts;

	// Add to stream packet. TODO use the appropriate function
	hdr->num_cuda_pkts++;
	memcpy(&rpkts[hdr->num_cuda_pkts - 1], packet, sizeof(cuda_packet_t));

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
	printd(DBG_DEBUG, "%s.%d: Send request header for %d packets.\n", __FUNCTION__, __LINE__, hdr->num_cuda_pkts);

	//send request header (including data size) for the batch of remote packets
	if (1 != put(myconn, hdr, sizeof(strm_hdr_t))) {
		retval = -1;
		goto finalize_do_cuda_rpc;		// @todo change it
	}

	// ASSUMPTION: request data and data size are properly populated before calling this enqueue
	// function
	//if(req_strm_has_data(&myconn->strm)) {
	if (hdr->data_size > 0) {
		assert(reqbuf && reqbuf_size);
		printd(DBG_DEBUG, "Send request data %d bytes. \n", hdr->data_size);
		// send the request data
		if (1 != put(myconn, (char *) myconn->request_data_buffer, hdr->data_size)) {
			retval = -1;
			goto finalize_do_cuda_rpc;
		}

		/*
		 printd(DBG_DEBUG, "REQUEST DATA SENT (%d bytes):\n", hdr->data_size);
		 for(i = 0; i < hdr->data_size; i++)
		 printd(DBG_DEBUG, "%c", myconn->request_data_buffer[i]);
		 printd(DBG_DEBUG,"\n");
		 */
	}
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

finalize_do_cuda_rpc:

	    memset(hdr, 0, sizeof(strm_hdr_t));
	    memset(rpkts, 0, MAX_REMOTE_BATCH_SIZE*sizeof(rpkt_t));

exit_do_cuda_rpc: conn_close(myconn);
	free(myconn);
	return retval;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// REMOTE FUNCTIONS (CLIENT)
//
// They should convert dom_info + cpkt -> remote packet stream
//
////////////////////////////////////////////////////////////////////////////////////////////////////
int nvbackGetDeviceCount_rpc(cuda_packet_t *packet){
    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    do_cuda_rpc(packet, NULL, 0, NULL, 0);
    return CUDA_SUCCESS;
}

int __nvback_cudaRegisterFatBinary_rpc(cuda_packet_t *packet) {
	printd(DBG_DEBUG, "%s: CUDA_ERROR=%d before RPC on method %d\n", __FUNCTION__,
			packet->ret_ex_val.err, packet->method_id);

	do_cuda_rpc(packet, (void *) packet->args[0].argui, packet->args[1].argi,
			NULL, 0);

	return CUDA_SUCCESS;
}



/////////////////////////
// SERVER SIDE CODE
/////////////////////////

int nvbackGetDeviceCount_srv(cuda_packet_t *packet, conn_t * pConn){
    int devCount = 0;

    // just call the function
    packet->ret_ex_val.err = cudaGetDeviceCount(&devCount);
    packet->args[0].argi = devCount;

    printd(DBG_DEBUG, "%s.%d: CUDA_ERROR=%p for method id=%d after calling method\n", __FUNCTION__, __LINE__, packet->ret_ex_val.handle, packet->method_id);
    return CUDA_SUCCESS;
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

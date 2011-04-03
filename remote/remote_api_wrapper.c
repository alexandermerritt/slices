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
#include "fatcubininfo.h"   // for fatcubin_info_t

//! Right now the host where we are connecting to (where clients, ie, *_rpc connects
//! to
#ifndef REMOTE_HOSTNAME
#	define REMOTE_HOSTNAME "cuda2.cc.gt.atl.ga.us"
#endif

//! open the connection forever; if you open and then close the
//! connection the program stops to work, so you have to make it static
static conn_t myconn;

//! stores information about the fatcubin_info
static fatcubin_info_t fatcubin_info;
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
/*int do_cuda_rpc(cuda_packet_t *packet,
		void *reqbuf,
		int reqbuf_size,
		void *rspbuf,
		int rspbuf_size) {


	return 0;
}*/

/**
 * closes the connection and frees the memory occupied by the connection
 * @param pConn (inout) The connection to be closed; changed to NULL
 * @param exit_code (in) What will be returned
 * @return exit_code indicates if we want this function to indicate the erroneous
 *        behaviour or not @see l_do_cuda_rpc
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
int l_do_cuda_rpc(cuda_packet_t *packet, void * reqbuf, const int reqbuf_size,
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
    l_do_cuda_rpc(packet, NULL, 0, NULL, 0);

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
    l_do_cuda_rpc(packet, NULL, 0, (void *) packet->args[0].argui, packet->args[2].argi);

    return (packet->ret_ex_val.err == 0) ? OK : ERROR;
}

int nvbackCudaFree_rpc(cuda_packet_t *packet){

    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    l_do_cuda_rpc(packet, NULL, 0, NULL, 0);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaMalloc_rpc(cuda_packet_t *packet){

    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);
    // clear the packet, we are also sending the size of
    // the memory to allocate
    packet->args[0].argp = NULL;
    l_do_cuda_rpc(packet, NULL, 0, NULL, 0);

    printd(DBG_DEBUG,"%s: devPtr is %p",__FUNCTION__, packet->args[0].argp);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaSetupArgument_rpc(cuda_packet_t *packet){
	printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
	            packet->ret_ex_val.err, packet->method_id);

	l_do_cuda_rpc(packet, (void *)packet->args[0].argp, packet->args[1].argi, NULL, 0);

	return (packet->ret_ex_val.err == 0) ? OK : ERROR;
}

int nvbackCudaConfigureCall_rpc(cuda_packet_t *packet){
    printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
            packet->ret_ex_val.err, packet->method_id);

    l_do_cuda_rpc(packet, NULL, 0, NULL, 0);
    return (packet->ret_ex_val.err == 0) ? OK : ERROR;
}

int nvbackCudaLaunch_rpc(cuda_packet_t * packet){

	printd(DBG_ERROR, "%s: FIXME!!!!!!!!!!!!! I am an ERRRRRRRRRRRRRRRRRRR%d\n",
				__FUNCTION__, packet->method_id);

	printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
	            packet->ret_ex_val.err, packet->method_id);

	l_do_cuda_rpc(packet,  NULL, 0, NULL, 0);


	return (packet->ret_ex_val.err == 0) ? OK : ERROR;
}

int nvbackCudaMemcpy_rpc(cuda_packet_t *packet){
	// this is the kind of the original cudaMemcpy
	int64_t kind = packet->args[3].argi;

	switch(packet->args[3].argi){
	case cudaMemcpyHostToDevice:
		packet->method_id = CUDA_MEMCPY_H2D;
		l_do_cuda_rpc(packet, (void *)packet->args[1].argui, packet->args[2].argi, NULL, 0);
		break;
	case cudaMemcpyDeviceToHost:
		packet->method_id = CUDA_MEMCPY_D2H;
		l_do_cuda_rpc(packet, NULL, 0, (void *)packet->args[0].argui, packet->args[2].argi);
		break;
	case cudaMemcpyDeviceToDevice:
		packet->method_id = CUDA_MEMCPY_D2D;
		packet->flags &= ~CUDA_Copytype;
		packet->flags &= ~CUDA_Addrshared;
		l_do_cuda_rpc(packet, NULL, 0, NULL, 0);
		break;
	case cudaMemcpyHostToHost:
		printd(DBG_ERROR, "Not implemented yet\n");
		return ERROR;

	default:
		printd(DBG_ERROR, "Unknown memcpy value %ld\n", kind);
		break;
	}

	return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int __nvback_cudaRegisterFatBinary_rpc(cuda_packet_t *packet) {
	printd(DBG_DEBUG, "%s: CUDA_ERROR=%d before RPC on method %d\n", __FUNCTION__,
			packet->ret_ex_val.err, packet->method_id);

	printd(DBG_DEBUG, "pPackedFat, packet->args[0].argui, size = %p, %ld\n",
				packet->args[0].argp, packet->args[1].argi);

	l_do_cuda_rpc(packet, (void *) packet->args[0].argui, packet->args[1].argi,
			NULL, 0);

	return (packet->ret_ex_val.handle != NULL) ? OK : ERROR;
}

int __nvback_cudaRegisterFunction_rpc(cuda_packet_t *packet) {
	printd(DBG_DEBUG, "%s: CUDA_ERROR=%d before RPC on method %d\n", __FUNCTION__,
				packet->ret_ex_val.err, packet->method_id);
	l_do_cuda_rpc(packet, (void *) packet->args[0].argui, packet->args[1].argi,
				NULL, 0);

	return (packet->ret_ex_val.err != 0) ? OK : ERROR;
}

int __nvback_cudaUnregisterFatBinary_rpc(cuda_packet_t *packet){
	printd(DBG_DEBUG, "CUDA_ERROR=%d before RPC on method %d\n",
	            packet->ret_ex_val.err, packet->method_id);
	l_do_cuda_rpc(packet, NULL, 0, NULL, 0);

	return (packet->ret_ex_val.err != 0) ? OK : ERROR;
}

/////////////////////////
// SERVER SIDE CODE
/////////////////////////

int nvbackCudaGetDeviceCount_srv(cuda_packet_t *packet, conn_t * pConn){
    int devCount = 0;

    // just call the function
    packet->ret_ex_val.err = cudaGetDeviceCount(&devCount);
    packet->args[0].argi = devCount;

    printd(DBG_DEBUG, "%s.%d: CUDA_ERROR=%d for method id=%d after calling method\n",
    		__FUNCTION__, __LINE__, packet->ret_ex_val.err, packet->method_id);
    return OK;
}

int nvbackCudaGetDeviceProperties_srv(cuda_packet_t *packet, conn_t * pConn){
	struct cudaDeviceProp * prop = (struct cudaDeviceProp *)pConn->response_data_buffer;

    pConn->response_data_size = sizeof(struct cudaDeviceProp);

    packet->ret_ex_val.err = cudaGetDeviceProperties(prop, packet->args[1].argi);

    // I guess you need to pack the change somehow the l_do_cuda_rpc
    // to use and send the response_data_buffer
    printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);

    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaMalloc_srv(cuda_packet_t * packet, conn_t * pConn){
	// just call the function
	packet->args[0].argp = NULL;

	packet->ret_ex_val.err = cudaMalloc(&packet->args[0].argp, packet->args[1].argi);

    printd(DBG_DEBUG,"%s: devPtr is %p, size %ld\n",__FUNCTION__,packet->args[0].argp, packet->args[1].argi);
    printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d after execution\n",
    		packet->ret_ex_val.err, packet->method_id);
	return OK;
}

int nvbackCudaFree_srv(cuda_packet_t *packet, conn_t *pConn){
	printd(DBG_DEBUG,"%s: devPtr is %p\n",__FUNCTION__,packet->args[0].argp);
    packet->ret_ex_val.err = cudaFree(packet->args[0].argp);
    printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaSetupArgument_srv(cuda_packet_t *packet, conn_t *pConn){
	void *arg = (void*) ((char *)pConn->request_data_buffer + packet->ret_ex_val.data_unit);
	packet->ret_ex_val.err = cudaSetupArgument( arg,
	            packet->args[1].argi,
	            packet->args[2].argi);
    printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaConfigureCall_srv(cuda_packet_t *packet, conn_t *pConn){
	packet->ret_ex_val.err = cudaConfigureCall( packet->args[0].arg_dim,
            packet->args[1].arg_dim,
            packet->args[2].argi,
            (cudaStream_t) packet->args[3].argi);

    printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);
    return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaLaunch_srv(cuda_packet_t * packet, conn_t * pConn){
	int i;
	const char *arg;

	printd(DBG_ERROR, "%s: ***************** FIXME ***** ERROR! ********%d\n",
			__FUNCTION__, packet->method_id);

	// this is entry for the cudaLaunch
	arg = (const char *)packet->args[0].argcp;

	packet->ret_ex_val.err = cudaErrorLaunchFailure;

	for(i = 0; i < fatcubin_info.num_reg_fns; ++i){
	  if (fatcubin_info.reg_fns[i] != NULL && fatcubin_info.reg_fns[i]->hostFEaddr == arg){
	      printd(DBG_DEBUG, "%s: function %p:%s\n", __FUNCTION__,
	    		  fatcubin_info.reg_fns[i]->hostFEaddr,
	    		  fatcubin_info.reg_fns[i]->hostFun);
	      packet->ret_ex_val.err = cudaLaunch(fatcubin_info.reg_fns[i]->hostFun);
	      break;
	  }
	}

	printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);

	return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

int nvbackCudaMemcpy_srv(cuda_packet_t *packet, conn_t * myconn){
	switch(packet->method_id){
	//case cudaMemcpyHostToHost:
	case CUDA_MEMCPY_H2H:
		// TODO: Should remote GPU handle this? - good question
		printd(DBG_WARNING, "Warning: CUDA_MEMCPY_H2H not supported\n");
		return ERROR;
	    //case cudaMemcpyHostToDevice:
	case CUDA_MEMCPY_H2D:
		printd(DBG_DEBUG, "request_data_size = %d, received count =%ld\n",
				myconn->request_data_size, packet->args[2].argi);
		assert(myconn->request_data_size == packet->args[2].argi);
		packet->args[1].argui = (uint64_t)((char *)myconn->request_data_buffer + packet->ret_ex_val.data_unit);
		break;
		//case cudaMemcpyDeviceToHost:
	case CUDA_MEMCPY_D2H:
		packet->args[0].argui = (uint64_t)myconn->response_data_buffer;
		myconn->response_data_size = packet->args[2].argi;
		//memset(myconn->response_data_buffer, 0, TOTAL_XFER_MAX);
		break;
		//case cudaMemcpyDeviceToHost:
	case CUDA_MEMCPY_D2D:
		// both src and dst addresses on device. nothing to modify
		break;
	}

	packet->ret_ex_val.err = cudaMemcpy( (void *)packet->args[0].argui,
	            (void *)packet->args[1].argui,
	            packet->args[2].argi,
	            packet->args[3].argi);

	printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);

	return (packet->ret_ex_val.err == 0)? OK : ERROR;
}

/**
 * in this function we do not return in the handle the
 * packet->ret_ex_val.err is not set I believe as
 * in most of the other calls so take this into account
 */
int __nvback_cudaRegisterFatBinary_srv(cuda_packet_t *packet, conn_t * myconn){

	__cudaFatCudaBinary * pFatC = malloc(sizeof(__cudaFatCudaBinary));
	void ** pFatCHandle;

	if( mallocCheck(pFatC, __FUNCTION__, NULL ) == ERROR ){
		exit(ERROR);
	}
	if( unpackFatBinary(pFatC, myconn->request_data_buffer) == ERROR ){
		printd(DBG_ERROR, "%s.%s: Problems with unpacking fat binary\n", __FILE__, __FUNCTION__);
		exit(ERROR);
	}
    //pFatC = copyFatBinary((__cudaFatCudaBinary *)((char *)myconn->request_data_buffer + packet->ret_ex_val.data_unit));

	// call __cudaRegisterFatBinary; otherwise the compiler complaints that
	// undefined reference to __cudaRegisterFatBinary
    pFatCHandle = __cudaRegisterFatBinary(pFatC);

    packet->args[1].argp = pFatC;
    packet->ret_ex_val.handle = pFatCHandle;

    printd(DBG_DEBUG, "%s: FATCUBIN HANDLE: registered %p\n", __FUNCTION__, pFatCHandle);

    return OK;
}

int __nvback_cudaRegisterFunction_srv(cuda_packet_t *packet, conn_t * myconn){
	reg_func_args_t * c_args = malloc(sizeof(__cudaFatCudaBinary));

	if( mallocCheck(c_args, __FUNCTION__, NULL ) == ERROR ){
			exit(ERROR);
	}

	if(	unpackRegFuncArgs(c_args, myconn->request_data_buffer) == ERROR ){
		printd(DBG_ERROR, "%s.%s: Problems with unpacking arguments in function register\n", __FILE__, __FUNCTION__);
		exit(ERROR);
	}

	printd(DBG_DEBUG, "%s:FATCUBIN HANDLE: received=%p, expected=%p",__FUNCTION__, c_args->fatCubinHandle, fatcubin_info.fatCubinHandle);
	assert(c_args->fatCubinHandle == fatcubin_info.fatCubinHandle);


	packet->args[2].argp = (void *)c_args;
	__cudaRegisterFunction( c_args->fatCubinHandle, (const char *)c_args->hostFun,
	                c_args->deviceFun, (const char *)c_args->deviceName,
	                c_args->thread_limit, c_args->tid,
	                c_args->bid, c_args->bDim, c_args->gDim, c_args->wSize);
//	dfi->reg_fns[dfi->num_reg_fns] = c_args;
//	dfi->num_reg_fns++;
	fatcubin_info.reg_fns[fatcubin_info.num_reg_fns] = c_args;
	fatcubin_info.num_reg_fns++;

	packet->ret_ex_val.err = 0;
	printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);
	return OK;
}

int __nvback_cudaUnregisterFatBinary_srv(cuda_packet_t *packet, conn_t  * myconn){

	int i;

	printd(DBG_DEBUG, "%s, FIX MEEEEEEEEEEEEEEEEEEEEEEEE!!!!!!!!!!!!!\n", __FUNCTION__);

	if( fatcubin_info.fatCubinHandle == NULL ){
		packet->ret_ex_val.err = OK;
		printd(DBG_DEBUG, "CUDA_ERROR=%p for method id=%d\n", packet->ret_ex_val.handle, packet->method_id);
		return ERROR;
	}
	__cudaUnregisterFatBinary(fatcubin_info.fatCubinHandle);

/*	for (i = 0; i < dfi->num_reg_vars; ++i)
	        freeRegVar(dfi->variables[i]);
	for (i = 0; i < dfi->num_reg_texs; ++i)
		freeRegTex(dfi->textures[i]); */
	for (i = 0; i < fatcubin_info.num_reg_fns; ++i)
		freeRegFunc(fatcubin_info.reg_fns[i]);
	/*for (i = 0; i < dfi->num_reg_shared; ++i) {
	   if (dfi->shared_vars[i] != NULL)
		   free(dfi->shared_vars[i]);
	   else
		   break;
	} */
	freeFatBinary(fatcubin_info.fatCubin);

	fatcubin_info.num_reg_fns = 0;
	fatcubin_info.num_reg_vars = 0;
	fatcubin_info.num_reg_texs = 0;
	fatcubin_info.num_reg_shared = 0;
	fatcubin_info.fatCubinHandle = NULL;

	printd(DBG_DEBUG, "CUDA_ERROR=%d for method id=%d\n", packet->ret_ex_val.err, packet->method_id);
	return OK;
}

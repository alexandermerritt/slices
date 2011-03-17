/**
 * @file remote_api_wrapper.h
 * @brief Constants for the network module, copied from
 * remote_gpu/network/remote_api_wrapper.c and
 * edited by me MS to make it more independent
 *
 * @date Feb 25, 2011
 * @author Magda Slawinska, magg@gatech.edu
 */

#ifndef __REMOTE_CUDA_CALLS_WRAPPER_H
#define __REMOTE_CUDA_CALLS_WRAPPER_H

#include "packetheader.h"
#include "connection.h"

extern void** __cudaRegisterFatBinary(void *fatCubin);

///////////////////
// RPC CALL UTILS//
///////////////////
int do_cuda_rpc( cuda_packet_t *packet,
                 void *request_buf,
                 int request_buf_size, 
                 void *response_buf, 
                 int response_buf_size);

/**
 * executes the cuda call over the network
 * @param pPacket the packet that contains data to be send and executed over the
 * network
 * @return OK everything went OK,
 *         ERROR if something went wrong
 */
int do_cuda_rpc1( cuda_packet_t *pPacket);

//////////////////////////
// CLIENT SIDE FUNCTIONS
// *_rpc means it marshalls the call and sends it to the remote host for execution.
// dom_info argument is void to fit into a jump table along with the functions from
// local_api_wrapper.c
//////////////////////////
int nvbackCudaMalloc_rpc(cuda_packet_t * packet);
int nvbackCudaFree_rpc(cuda_packet_t * packet);
int nvbackCudaGetDeviceCount_rpc(cuda_packet_t *packet);
int nvbackCudaGetDeviceProperties_rpc(cuda_packet_t *packet);
int __nvback_cudaRegisterFatBinary_rpc(cuda_packet_t *packet);

/////////////////////////
// SERVER SIDE CODE
/////////////////////////
int nvbackCudaMalloc_srv(cuda_packet_t * packet, conn_t * pConn);
int nvbackCudaGetDeviceCount_srv(cuda_packet_t * packet, conn_t * pConn);
int nvbackCudaGetDeviceProperties_srv(cuda_packet_t * packet, conn_t *pConn);
int nvbackCudaFree_srv(cuda_packet_t * packet, conn_t * pConn);
int __nvback_cudaRegisterFatBinary_srv(cuda_packet_t *packet, conn_t * myconn);

#endif

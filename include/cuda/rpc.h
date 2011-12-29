/**
 * @file rpc.h
 * @date 2011-12-28
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief RPC connection structure for CUDA packets.
 */

#ifndef CUDA_RPC_H
#define CUDA_RPC_H

// Project includes
#include <cuda/packet.h>
#include <io/sock.h>

/**
 * State maintained with each remote connection across which CUDA RPC packets
 * are sent. Batching is provided to decrease the number of times the network is
 * used, increasing the amount of data sent with each.
 */
struct cuda_rpc
{
	struct cuda_pkt_batch batch;
	struct sockconn sockconn;
};

int cuda_rpc_init(struct cuda_rpc *rpc);
int cuda_rpc_connect(struct cuda_rpc *rpc, const char *ip, const char *port);
int cuda_rpc_close(struct cuda_rpc *rpc);

#endif	/* CUDA_RPC_H */

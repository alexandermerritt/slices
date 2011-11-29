/**
 * @file include/cuda/ops.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief This file defines a common function prototype we may use in jump
 * tables for sending a CUDA RPC along a local or remote or some other form of
 * data path. Refer to assembly.c:demultiplex_call for examples on how these
 * functions are invoked, and common/cuda/execute.c for example implementations.
 */
#ifndef _CUDA_OPS_H
#define _CUDA_OPS_H

// Project includes
#include <packetheader.h>

/**
 * This defines the prototype for similarly defined functions implementing the
 * CUDA API across the system. Argument ordering is important.
 *
 * Execute packet locally
 *		int NAME(struct cuda_packet*);
 *		int NAME(struct cuda_packet*, struct fatcubins*);
 *
 * Execute packet via RPC
 * 		int NAME(struct cuda_packet*, NULL,              struct cuda_rpc_conn*);
 * 		int NAME(struct cuda_packet*, struct fatcubins*, struct cuda_rpc_conn*);
 *
 * Arguments after the cuda packet will need to be looked up via use of a
 * va_list within the function itself that requires it.
 */
#define OPS_FN_PROTO(name)		int name (struct cuda_packet *pkt, ...)
#define OPS_FN_PROTO_PTR(name)	int (* name)(struct cuda_packet*, ...)

struct cuda_ops
{
	OPS_FN_PROTO_PTR(registerFatBinary);
	OPS_FN_PROTO_PTR(unregisterFatBinary);
	OPS_FN_PROTO_PTR(registerFunction);
	OPS_FN_PROTO_PTR(registerVar);
	OPS_FN_PROTO_PTR(setDevice);
	OPS_FN_PROTO_PTR(configureCall);
	OPS_FN_PROTO_PTR(setupArgument);
	OPS_FN_PROTO_PTR(launch);
	OPS_FN_PROTO_PTR(threadExit);
	OPS_FN_PROTO_PTR(threadSynchronize);
	OPS_FN_PROTO_PTR(malloc);
	OPS_FN_PROTO_PTR(free);
	OPS_FN_PROTO_PTR(memcpyH2D);
	OPS_FN_PROTO_PTR(memcpyD2H);
	OPS_FN_PROTO_PTR(memcpyD2D);
	OPS_FN_PROTO_PTR(memcpyToSymbolH2D);
	OPS_FN_PROTO_PTR(memcpyFromSymbolD2H);
};

extern const struct cuda_ops exec_ops;
extern const struct cuda_ops rpc_ops;

#endif	/* _CUDA_OPS_H */

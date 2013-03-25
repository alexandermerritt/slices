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
#include <cuda/packet.h>

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
	// accept cuda_packet
	OPS_FN_PROTO_PTR(bindTexture);
	OPS_FN_PROTO_PTR(configureCall);
	OPS_FN_PROTO_PTR(createChannelDesc);
	OPS_FN_PROTO_PTR(free);
	OPS_FN_PROTO_PTR(freeArray);
	OPS_FN_PROTO_PTR(freeHost);
	OPS_FN_PROTO_PTR(funcGetAttributes);
	OPS_FN_PROTO_PTR(getMemInfo);
	OPS_FN_PROTO_PTR(getTextureReference);
	OPS_FN_PROTO_PTR(hostAlloc);

	OPS_FN_PROTO_PTR(eventCreate);
	OPS_FN_PROTO_PTR(eventCreateWithFlags);
	OPS_FN_PROTO_PTR(eventDestroy);
	OPS_FN_PROTO_PTR(eventElapsedTime);
	OPS_FN_PROTO_PTR(eventQuery);
	OPS_FN_PROTO_PTR(eventRecord);
	OPS_FN_PROTO_PTR(eventSynchronize);

	OPS_FN_PROTO_PTR(malloc);
	OPS_FN_PROTO_PTR(mallocArray);
	OPS_FN_PROTO_PTR(mallocPitch);
	OPS_FN_PROTO_PTR(memcpyAsyncD2D);
	OPS_FN_PROTO_PTR(memcpyAsyncD2H);
	OPS_FN_PROTO_PTR(memcpyAsyncH2D);
	OPS_FN_PROTO_PTR(memcpyD2D);
	OPS_FN_PROTO_PTR(memcpyD2H);
	OPS_FN_PROTO_PTR(memcpyH2D);
	OPS_FN_PROTO_PTR(memcpyToArrayD2D);
	OPS_FN_PROTO_PTR(memcpyToArrayH2D);
	OPS_FN_PROTO_PTR(memGetInfo);
	OPS_FN_PROTO_PTR(memset);
	OPS_FN_PROTO_PTR(registerTexture);
	OPS_FN_PROTO_PTR(setDevice);
	OPS_FN_PROTO_PTR(setDeviceFlags);
	OPS_FN_PROTO_PTR(setupArgument);
	OPS_FN_PROTO_PTR(setValidDevices);
	OPS_FN_PROTO_PTR(streamCreate);
	OPS_FN_PROTO_PTR(streamDestroy);
	OPS_FN_PROTO_PTR(streamQuery);
	OPS_FN_PROTO_PTR(streamSynchronize);
	OPS_FN_PROTO_PTR(threadExit);
	OPS_FN_PROTO_PTR(threadSynchronize);
	OPS_FN_PROTO_PTR(unregisterFatBinary);

	// accept cuda_packet and fatcubins
	OPS_FN_PROTO_PTR(bindTextureToArray);
	OPS_FN_PROTO_PTR(launch);
	OPS_FN_PROTO_PTR(memcpyFromSymbolD2H);
	OPS_FN_PROTO_PTR(memcpyToSymbolAsyncH2D);
	OPS_FN_PROTO_PTR(memcpyToSymbolH2D);
	OPS_FN_PROTO_PTR(registerFatBinary);
	OPS_FN_PROTO_PTR(registerFunction);
	OPS_FN_PROTO_PTR(registerVar);
};

extern const struct cuda_ops exec_ops;
extern struct cuda_ops rpc_ops;

#endif	/* _CUDA_OPS_H */

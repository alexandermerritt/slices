/**
 * @file common/cuda/rpc.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief Send serialized CUDA packets across the network for another machine to
 * execute. Analogous to execute.c, but requiring additional state describing a
 * persistent connection.
 */

// System includes
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>

// Project includes
#include <cuda/fatcubininfo.h>
#include <cuda/hidden.h>
#include <cuda/method_id.h>
#include <cuda/ops.h>
#include <cuda/packet.h>
#include <cuda/rpc.h>
#include <debug.h>
#include <io/sock.h>
#include <util/compiler.h>
#include <util/timer.h>

/* NOTES
 *
 * Please read the notes in execute.c in addition to these.
 */

/*-------------------------------------- PUBLIC FUNCTIONS --------------------*/

int cuda_rpc_init(struct cuda_rpc *rpc, size_t batch_size)
{
	memset(rpc, 0, sizeof(*rpc));
	rpc->batch.buffer = malloc(CUDA_BATCH_BUFFER_SZ);
	if (!rpc->batch.buffer) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		return -1;
	}
	if (batch_size > CUDA_BATCH_MAX) {
		printd(DBG_ERROR, "Batch size %lu too large (max %d)\n",
				batch_size, CUDA_BATCH_MAX);
		return -1;
	}
	rpc->batch.max = batch_size;
#if 0
	int err = mlock(rpc->batch.buffer, CUDA_BATCH_BUFFER_SZ);
	if (err < 0) {
		printd(DBG_WARNING, "Could not pin batch buffer: %s\n",
				strerror(errno));
	}
#endif
	return 0;
}

int cuda_rpc_connect(struct cuda_rpc *rpc, const char *ip, const char *port)
{
#if defined(NIC_SDP)
	return conn_connect(&rpc->sockconn, ip, port, true);
#elif defined(NIC_ETHERNET)
	return conn_connect(&rpc->sockconn, ip, port, false);
#else
#error NIC_* not defined
#endif
}

int cuda_rpc_close(struct cuda_rpc *rpc)
{
	int err;
	err = conn_close(&rpc->sockconn);
	if (err < 0)
		return -ENETDOWN;
	if (rpc->batch.buffer)
		free(rpc->batch.buffer);
	return 0;
}

/*-------------------------------------- HIDDEN FUNCTIONS --------------------*/

#define FAIL_ON_CONN_ERR(err)		\
	do { 							\
		if (unlikely(err <= 0)) {	\
			exit_errno = -ENETDOWN;	\
			goto fail;				\
		}							\
	} while(0)

static int
batch_deliver(struct cuda_rpc *rpc, struct cuda_packet *return_pkt)
{
	int exit_errno;
	struct cuda_pkt_batch *batch = &rpc->batch;
	size_t payload_len = 0UL;

	printd(DBG_INFO, "pkts = %lu size = %lu\n",
			batch->header.num_pkts, batch->header.bytes_used);

	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, &batch->header, sizeof(batch->header)) );
#if defined(NIC_SDP)
	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, batch->buffer, batch->header.bytes_used + ZCPY_TRIGGER_SZ) );
#else
	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, batch->buffer, batch->header.bytes_used) );
#endif

#ifndef NO_PIPELINING
    struct cuda_packet *last_pkt = NULL;
    last_pkt = (struct cuda_packet*)
        ((uintptr_t)batch->buffer +
         (intptr_t)batch->header.offsets[batch->header.num_pkts - 1]);
    if (last_pkt->is_sync) { /* only expect a return packet if last is sync */
#endif

#if defined(NIC_SDP)
	    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, return_pkt, sizeof(*return_pkt) + ZCPY_TRIGGER_SZ) );
#else
	    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, return_pkt, sizeof(*return_pkt)) );
#endif

	    payload_len = return_pkt->len - sizeof(*return_pkt);
	    if (payload_len > 0) {
#if defined(NIC_SDP)
		    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, (return_pkt + 1), payload_len + ZCPY_TRIGGER_SZ) );
#else
		    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, (return_pkt + 1), payload_len) );
#endif
	    }
#ifndef NO_PIPELINING
    }
#endif

	return 0;
fail:
	return exit_errno;
}

static void
batch_clear(struct cuda_pkt_batch *batch)
{
	batch->header.num_pkts = 0UL;
	batch->header.bytes_used = 0UL;
	// don't free buffer storage
}

static int
batch_append_and_flush(struct cuda_rpc *rpc, struct cuda_packet *pkt)
{
	// We append unless we do not have space or the packet holds state for a
	// synchronous function call

	int err = 0;
	struct cuda_pkt_batch *batch = &rpc->batch;
	size_t rpc_size = pkt->len; // len includes size of struct cuda_packet
	uintptr_t buf_ptr = (uintptr_t)batch->buffer + batch->header.bytes_used;

	printd(DBG_DEBUG, "pkt %lu offset %lu len %lu\n",
			batch->header.num_pkts, batch->header.bytes_used, rpc_size);

	batch->header.offsets[batch->header.num_pkts++] = batch->header.bytes_used;

	// We assume we will always have storage space to hold a packet and its
	// data, and that a "flush" will only occur due to a synchronous packet or
	// we run out of slots to hold packets. Thus, we batch as aggressively as we
	// can.
	size_t remaining_storage = CUDA_BATCH_BUFFER_SZ - batch->header.bytes_used;
	BUG(remaining_storage < rpc_size);
	BUG(batch->header.num_pkts > CUDA_BATCH_MAX);
	BUG(!batch->buffer);

	memcpy((void*)buf_ptr, pkt, rpc_size);
	batch->header.bytes_used += rpc_size;

	if (pkt->is_sync || (batch->header.num_pkts >= batch->max)) {
		printd(DBG_INFO, "\t--> flushing\n");
		err = batch_deliver(rpc, pkt);
		batch_clear(batch);
	} else {
		pkt->ret_ex_val.err = cudaSuccess;
	}

	return err;
}

/**
 * Pull out the network connection structure from the va_list given to the
 * function. It is found second in the va_list.
 *
 * @param rpc		Pointer to struct cuda_rpc*
 * @param argname	Name of last named parameter of function. Refer to
 * 					OPS_FN_PROTO.
 */
#define GET_CONN_VALIST(rpc,argname)					\
	do {												\
		va_list extra;									\
		va_start(extra, argname);						\
		va_arg(extra, void*);	/* skip first arg */	\
		(rpc) = va_arg(extra, struct cuda_rpc*);		\
		BUG(!(rpc));									\
		va_end(extra);									\
	} while(0)

#define FAIL_ON_BATCH_ERR(func)		\
	do { 							\
		if (unlikely((func) < 0)) {	\
			exit_errno = -ENETDOWN;	\
			goto fail;				\
		}							\
	} while(0)

/**
 * This function requires an additional argument beyond those found in
 * cuda/execute.c; it is found at the _second_ argument in the va_list (so third
 * function argument), ignoring the first:
 *
 * 		struct cuda_rpc*
 *
 * Each RPC is queued into a separate structure called a 'batch'. Metadata
 * maintains offsets for each packet into the batch, and when a synchronous
 * function is encountered, the entire batch is sent to the remote node to
 * reduce network overheads.
 *
 * As each packet contains a length member, we needn't examine each function
 * individually to calculate the size of the payload associated with the RPC, as
 * this is done for us in the interposing library.
 */
static OPS_FN_PROTO(CudaDoRPC)
{
	int exit_errno;
	struct cuda_rpc *rpc;
	GET_CONN_VALIST(rpc,pkt);
	FAIL_ON_BATCH_ERR( batch_append_and_flush(rpc, pkt) );
	return 0;
fail:
	return exit_errno;
}

const struct cuda_ops rpc_ops =
{
	.setupArgument			= CudaDoRPC,
	.configureCall			= CudaDoRPC,
	.launch					= CudaDoRPC,
	.freeArray				= CudaDoRPC,
	.free					= CudaDoRPC,
	//.freeHost				= CudaDoRPC,
	.funcGetAttributes		= CudaDoRPC,
	.malloc					= CudaDoRPC,
	.threadSynchronize		= CudaDoRPC,
	.mallocPitch			= CudaDoRPC,
	.memcpyD2D				= CudaDoRPC,
	.memcpyD2H				= CudaDoRPC,
	.memcpyFromSymbolD2H	= CudaDoRPC,
	.memcpyH2D				= CudaDoRPC,
	.memcpyToSymbolH2D		= CudaDoRPC,
	.memcpyToSymbolAsyncH2D	= CudaDoRPC,
	.memset					= CudaDoRPC,
	.memGetInfo				= CudaDoRPC,
	.registerFatBinary		= CudaDoRPC,
	.registerFunction		= CudaDoRPC,
	.registerVar			= CudaDoRPC,
	.setDevice				= CudaDoRPC,
	.setDeviceFlags			= CudaDoRPC,
	.setValidDevices		= CudaDoRPC,
	.streamCreate			= CudaDoRPC,
	.streamDestroy			= CudaDoRPC,
	.streamQuery			= CudaDoRPC,
	.streamSynchronize		= CudaDoRPC,
	.threadExit				= CudaDoRPC,
	.unregisterFatBinary	= CudaDoRPC
};

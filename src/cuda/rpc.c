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

static inline struct cuda_packet *
last_pkt(struct cuda_rpc *rpc)
{
    struct cuda_pkt_batch *b = &rpc->batch;
    return (struct cuda_packet*)((uintptr_t)b->buffer +
            (uintptr_t)b->header.offsets[b->header.num_pkts - 1]);
}

static void
batch_clear(struct cuda_pkt_batch *batch)
{
	batch->header.num_pkts = 0UL;
	batch->header.bytes_used = 0UL;
	// don't free buffer storage
}

struct flush
{
    struct timespec ts;
    size_t bytes;
    unsigned long lat, exec; /* not used if blocking false */
    bool blocking, has_ret_payload;
};

#define MAX_HISTORY (8UL << 20)
unsigned long num_flushes = 0UL;
struct flush flushes[MAX_HISTORY];

void dump_flushes(void)
{
    char filename[256];
    unsigned long n = 0;
    struct flush *f;
    FILE *fp;

    unsigned long v;
    struct timer t;
    timer_init(CLOCK_REALTIME, &t);
    timer_start(&t);

    snprintf(filename, 256, "/lustre/medusa/merritt/dump.%d", getpid());
    fp = fopen(filename, "w");
    if (!fp)
        return;
    fprintf(fp, "bytes latency synchrony time execlat\n");
    while (n < num_flushes) {
        f = &flushes[n];
        fprintf(fp, "%lu %lu %d %lu %lu\n", f->bytes, f->lat, f->blocking,
                f->ts.tv_sec * 1000000000UL + f->ts.tv_nsec, f->exec);
        n++;
    }
    fclose(fp);

    v = timer_end(&t, MICROSECONDS);
    printf("> %s %lu usec\n", __func__, v);
}

static int
batch_deliver(struct cuda_rpc *rpc, struct cuda_packet *return_pkt)
{
	int exit_errno;
	struct cuda_pkt_batch *batch = &rpc->batch;
	size_t payload_len = 0UL;
    struct flush *f;
    struct timer t;

	printd(DBG_INFO, "pkts = %lu size = %lu\n",
			batch->header.num_pkts, batch->header.bytes_used);

    timer_init(CLOCK_REALTIME, &t);

    f = &flushes[num_flushes];
    clock_gettime(CLOCK_REALTIME, &f->ts);
    f->bytes = sizeof(batch->header);
    f->blocking = false;
	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, &batch->header, sizeof(batch->header)) );

#if defined(NIC_SDP)
    f = &flushes[++num_flushes];
    clock_gettime(CLOCK_REALTIME, &f->ts);
    f->bytes = batch->header.bytes_used + ZCPY_TRIGGER_SZ;
    timer_start(&t); // ignored if batch is non-blocking
	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, batch->buffer, batch->header.bytes_used + ZCPY_TRIGGER_SZ) );
#else
	FAIL_ON_CONN_ERR( conn_put(&rpc->sockconn, batch->buffer, batch->header.bytes_used) );
#endif

#ifndef NO_PIPELINING
    if (last_pkt(rpc)->is_sync) { /* only expect a return packet if last is sync */
#endif

#if defined(NIC_SDP)
        f->blocking = true;
        f->bytes += sizeof(*return_pkt) + ZCPY_TRIGGER_SZ;
	    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, return_pkt, sizeof(*return_pkt) + ZCPY_TRIGGER_SZ) );
#else
	    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, return_pkt, sizeof(*return_pkt)) );
#endif

	    payload_len = return_pkt->len - sizeof(*return_pkt);
	    if (payload_len > 0) {
#if defined(NIC_SDP)
            f->bytes += payload_len + ZCPY_TRIGGER_SZ;
            f->has_ret_payload = true;
		    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, (return_pkt + 1), payload_len + ZCPY_TRIGGER_SZ) );
#else
		    FAIL_ON_CONN_ERR( conn_get(&rpc->sockconn, (return_pkt + 1), payload_len) );
#endif
	    }
#ifndef NO_PIPELINING
        f->lat = timer_end(&t, MICROSECONDS);
        f->exec = return_pkt->execlat;
    }
#endif
    ++num_flushes;
	batch_clear(batch);
	return 0;
fail:
	return exit_errno;
}

static int
batch_append_and_flush(struct cuda_rpc *rpc, struct cuda_packet *pkt)
{
	// We append unless we do not have space or the packet holds state for a
	// synchronous function call

    int err = 0;
	struct cuda_pkt_batch *batch = &rpc->batch;
	size_t rpc_size = pkt->len; // len includes size of struct cuda_packet
	uintptr_t buf_ptr = 0;
	size_t remaining;

    // if no space for incoming pkt, flush first
    remaining = (CUDA_BATCH_BUFFER_SZ - batch->header.bytes_used);
    if (remaining < rpc_size) {
        printd(DBG_DEBUG, "pre-flushing!\n");
        BUG(last_pkt(rpc)->is_sync); // should have already been flushed
        batch_deliver(rpc, pkt);
    }

	printd(DBG_DEBUG, "pkt %lu offset %lu len %lu\n",
			batch->header.num_pkts, batch->header.bytes_used, rpc_size);

	buf_ptr = (uintptr_t)batch->buffer + batch->header.bytes_used;
	batch->header.offsets[batch->header.num_pkts++] = batch->header.bytes_used;

	// We assume we will always have storage space to hold a packet and its
	// data, and that a "flush" will only occur due to a synchronous packet or
	// we run out of slots to hold packets. Thus, we batch as aggressively as we
	// can.
	remaining = CUDA_BATCH_BUFFER_SZ - batch->header.bytes_used;
	BUG(remaining < rpc_size); // true if a single memcpy moves more than size of buffer
	BUG(batch->header.num_pkts > CUDA_BATCH_MAX);
	BUG(!batch->buffer);

	memcpy((void*)buf_ptr, pkt, rpc_size);
	batch->header.bytes_used += rpc_size;

	if (pkt->is_sync || (batch->header.num_pkts >= batch->max)) {
		printd(DBG_INFO, "\t--> flushing\n");
        // XXX should the return pkt reside at buf[0] instead?
		err = batch_deliver(rpc, pkt/*return pkt*/);
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

struct cuda_ops rpc_ops;

// XXX HACK
//
// Need to set all members of rpc_ops to same function, but because it is a
// struct, there doesn't seem to be a clean compile-time method for this. Even
// no clean runtime method :) It's problematic when new functions are added to
// struct rpc_ops and are not updated in either exec_ops or rpc_ops. This is a
// dynamic library constructor which is called once when libsfcuda.so is loaded,
// allowing new members to be added to point to the same function without having
// to change this code.
//
// TODO Since all rpc_ops point to the same function, I might as well just not
// use the structure... It was around originally because we used a generic
// rpc_ops pointer depending on local/remote vGPU state to avoid knowing which
// it was underneath.
//
// This assumes ALL members in rpc_ops are of the SAME TYPE/SIZE.
__attribute__((constructor)) void init_rpc_ops(void)
{
    typedef OPS_FN_PROTO_PTR(ops_t);
    ops_t *ops_ptr = (ops_t*) &rpc_ops;
    int count = sizeof(struct cuda_ops) / sizeof(ops_t);
    while (--count >= 0)
        *ops_ptr++ = CudaDoRPC;
}

#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Stream Management
//===----------------------------------------------------------------------===//

// XXX NOTE XXX
//
// In v3.2 cudaStream_t is defined: typedef struct CUstream_st* cudaStream_t.
// So this is a pointer to a pointer allocated by the caller, and is meant as an
// output parameter. The compiler and debugger cannot tell me what this
// structure looks like, nor what its size is, so we'll have to maintain a cache
// of cudaStream_t in the sink processes, and return our own means of
// identification for them to the caller instead.  I'm guessing that the Runtime
// API allocates some kind of memory or something to this pointer internally,
// and uses this pointer as a key to look it up. We'll have to do the same.
//
// cuda v3.2 cudaStream_t: pointer to opaque struct
// cuda v2.3 cudaStream_t: integer

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaStreamCreate(pStream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaStreamCreate(pStream, lat);
#endif

	printd(DBG_DEBUG, "stream=%p\n", *pStream);

	update_latencies(lat, CUDA_STREAM_CREATE);
	return cerr;
}

#if 0
cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet struct cuda_packet;
	memset(&tpkt, 0, sizeof(struct cuda_packet));
	cerr = bypass.cudaStreamDestroy(stream);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt);
	shmpkt = &tpkt;
#else
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	pack_cudaStreamDestroy(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_STREAM_DESTROY);
	return cerr;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "stream=%p\n", stream);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt;
	memset(&tpkt, 0, sizeof(tpkt));
	cerr = bypass.cudaStreamQuery(stream);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt);
	shmpkt = &tpkt;
#else
	pack_cudaStreamQuery(shmpkt, stream);
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_STREAM_QUERY);
	return cerr;
}
#endif

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "stream=%p\n", stream);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaStreamSynchronize(stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaStreamSynchronize(stream, lat);
#endif

	update_latencies(lat, CUDA_STREAM_SYNCHRONIZE);
	return cerr;
}


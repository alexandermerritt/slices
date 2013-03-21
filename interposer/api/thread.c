#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Thread Management [DEPRECATED]
//===----------------------------------------------------------------------===//

cudaError_t cudaThreadExit(void)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadExit();
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaThreadExit(lat);
#endif

	update_latencies(lat, CUDA_THREAD_EXIT);
	return cerr;
}

cudaError_t cudaThreadSynchronize(void)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);
    trace_timestamp();
	
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadSynchronize();
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaThreadSynchronize(lat);
#endif

	update_latencies(lat, CUDA_THREAD_SYNCHRONIZE);
	return cerr;
}

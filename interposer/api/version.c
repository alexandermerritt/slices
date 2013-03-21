#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Version Management
//===----------------------------------------------------------------------===//

cudaError_t cudaDriverGetVersion(int *driverVersion)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaDriverGetVersion(driverVersion);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_DRIVER_GET_VERSION);
	return cerr;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaRuntimeGetVersion(runtimeVersion);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_RUNTIME_GET_VERSION);
	return cerr;
}


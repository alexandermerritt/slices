#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Device Management
//===----------------------------------------------------------------------===//

cudaError_t cudaGetDevice(int *device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "thread=%lu\n", pthread_self());
    LAT_DECLARE(lat);
    trace_timestamp();
	
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDevice(device);
	TIMER_END(t, lat->exec.call);
    lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaGetDevice(device, lat);
#endif

	update_latencies(lat, CUDA_GET_DEVICE);
	return cerr;
}

cudaError_t cudaGetDeviceCount(int *count)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDeviceCount(count);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaGetDeviceCount(count, lat);
	printd(DBG_DEBUG, "%d\n", *count);
#endif

	update_latencies(lat, CUDA_GET_DEVICE_COUNT);
	return cerr;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "dev=%d\n", device);
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaGetDeviceProperties(prop,device);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + sizeof(*prop);
#else
	cerr = assm_cudaGetDeviceProperties(prop, device, lat);
#endif

	update_latencies(lat, CUDA_GET_DEVICE_PROPERTIES);
	return cerr;
}

cudaError_t cudaSetDevice(int device)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "device=%d\n", device);
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetDevice(device);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaSetDevice(device, lat);
#endif

	update_latencies(lat, CUDA_SET_DEVICE);
	return cerr;
}

cudaError_t cudaSetDeviceFlags(unsigned int flags)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "flags=0x%x\n", flags);
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetDeviceFlags(flags);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaSetDeviceFlags(flags, lat);
#endif

	update_latencies(lat, CUDA_SET_DEVICE_FLAGS);
	return cerr;
}

cudaError_t cudaSetValidDevices(int *device_arr, int len)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called; IGNORING\n");
    LAT_DECLARE(lat);
    trace_timestamp();

    /* XXX Ignore this stupid CUDA function */
    return cudaSuccess;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaThreadExit();
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + (len * sizeof(*device_arr));
#else
	cerr = assm_cudaSetValidDevices(device_arr, len, lat);
#endif

	update_latencies(lat, CUDA_SET_VALID_DEVICES);
	return cerr;
}


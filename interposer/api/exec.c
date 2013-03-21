#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Execution Control
//===----------------------------------------------------------------------===//

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "grid={%d,%d,%d} block={%d,%d,%d} shmem=%lu strm=%p\n",
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z,
			sharedMem, stream);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaConfigureCall(gridDim,blockDim,sharedMem,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaConfigureCall(gridDim, blockDim, sharedMem, stream, lat);
#endif

	update_latencies(lat, CUDA_CONFIGURE_CALL);
	return cerr;
}

#if 0
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *func)
{
	struct cuda_packet *shmpkt;
	cudaError_t cerr;
	TIMER_DECLARE1(tsetup);

	printd(DBG_DEBUG, "func='%s'\n", func);

	if (!attr || !func) {
		return cudaErrorInvalidDeviceFunction;
	}

	TIMER_START(tsetup);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt;
	memset(&tpkt, 0, sizeof(tpkt));
	cerr = bypass.cudaFuncGetAttributes(attr,func);
	TIMER_END(tsetup, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt) + strlen(func) + 1;
	shmpkt = &tpkt;
#else
	void *func_shm = NULL; // Pointer to func str within shm
	void *attr_shm = NULL; // Pointer to attr within shm
	unsigned int func_len;
	TIMER_DECLARE1(twait);

	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	func_len = strlen(func) + 1;
	shmpkt->method_id = CUDA_FUNC_GET_ATTR;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argull = sizeof(*shmpkt); // offset
	shmpkt->args[1].argull = (shmpkt->args[0].argull + sizeof(*attr)); // offset
	shmpkt->args[2].arr_argi[0] = func_len;
	func_shm = (void*)((uintptr_t)shmpkt + shmpkt->args[1].argull);
	memcpy(func_shm, func, func_len);
	shmpkt->len = sizeof(*shmpkt) + func_len;
	shmpkt->is_sync = true;
	TIMER_PAUSE(tsetup);

	TIMER_START(twait);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(twait, shmpkt->lat.lib.wait);

	TIMER_RESUME(tsetup);
	// Copy out the structure into the user argument
	attr_shm = (struct cudaFuncAttributes*)((uintptr_t)shmpkt + shmpkt->args[0].argull);
	memcpy(attr, attr_shm, sizeof(*attr));
	TIMER_END(tsetup, shmpkt->lat.lib.setup);

	cerr = shmpkt->ret_ex_val.err;
#endif

	update_latencies(lat, CUDA_FUNC_GET_ATTR);
	return cerr;
}
#endif

cudaError_t cudaLaunch(const char *entry)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();
	printd(DBG_DEBUG, "entry=%p\n", entry);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaLaunch(entry);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaLaunch(entry, lat);
#endif

	update_latencies(lat, CUDA_LAUNCH);
	return cerr;
}

cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "arg=%p size=%lu offset=%lu\n",
			arg, size, offset);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaSetupArgument(arg,size,offset);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + size;
#else
	cerr = assm_cudaSetupArgument(arg, size, offset, lat);
#endif

	update_latencies(lat, CUDA_SETUP_ARGUMENT);
	return cerr;
}


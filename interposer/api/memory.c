#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Memory Management
//===----------------------------------------------------------------------===//

cudaError_t cudaFree(void * devPtr)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();
	printd(DBG_DEBUG, "devPtr=%p\n", devPtr);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFree(devPtr);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaFree(devPtr, lat);
#endif

	update_latencies(lat, CUDA_FREE);
	return cerr;
}

cudaError_t cudaFreeArray(struct cudaArray * array)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();
	printd(DBG_DEBUG, "array=%p\n", array);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFreeArray(array);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaFreeArray(array, lat);
#endif

	update_latencies(lat, CUDA_FREE_ARRAY);
	return cerr;
}

cudaError_t cudaFreeHost(void * ptr)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaFreeHost(ptr);
	TIMER_END(t, lat->exec.call);
#else
	cerr = assm_cudaFreeHost(ptr, lat);
#endif

	update_latencies(lat, CUDA_FREE_HOST);
	return cerr;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaHostAlloc(pHost,size,flags);
	TIMER_END(t, lat->exec.call);
#else
	cerr = assm_cudaHostAlloc(pHost, size, flags, lat);
#endif

	printd(DBG_DEBUG, "host=%p size=%lu flags=0x%x (ignored)\n",
            *pHost, size, flags);
	update_latencies(lat, CUDA_HOST_ALLOC);
	return cerr;
}

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMalloc(devPtr,size);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaMalloc(devPtr, size, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p size=%lu cerr=%d\n",
            *devPtr, size, cerr);
	update_latencies(lat, CUDA_MALLOC);
	return cerr;
}

cudaError_t cudaMallocArray(
		struct cudaArray **array, // stores a device address
		const struct cudaChannelFormatDesc *desc,
		size_t width, size_t height, unsigned int flags)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMallocArray(array,desc,width,height,flags);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaMallocArray(array, desc, width, height, flags, lat);
#endif

	printd(DBG_DEBUG, "array=%p, desc=%p width=%lu height=%lu flags=0x%x\n",
			*array, desc, width, height, flags);
	update_latencies(lat, CUDA_MALLOC_ARRAY);
	return cerr;
}

cudaError_t cudaMallocPitch(
		void **devPtr, size_t *pitch, size_t width, size_t height) {
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMallocPitch(devPtr,pitch,width,height);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaMallocPitch(devPtr, pitch, width, height, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p pitch=%lu\n", *devPtr, *pitch);
	update_latencies(lat, CUDA_MALLOC_PITCH);
	return cerr;
}

cudaError_t cudaMemcpy(void *dst, const void *src,
		size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d\n",
			dst, src, count, kind);

	method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToHost: id = CUDA_MEMCPY_H2H; break;
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_H2D; break;
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_D2D; break;
        default:
            fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
            abort();
	}

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpy(dst,src,count,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpy(dst, src, count, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
		enum cudaMemcpyKind kind, cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "dst=%p src=%p count=%lu kind=%d stream=%p\n",
			dst, src, count, kind, stream);

	method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToHost: id = CUDA_MEMCPY_ASYNC_H2H; break;
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_ASYNC_H2D; break;
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_ASYNC_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_ASYNC_D2D; break;
        default:
            fprintf(stderr, "> %s kind %d unhandled\n", __func__, kind);
            abort();
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyAsync(dst,src,count,kind,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpyAsync(dst, src, count, kind, stream, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyFromSymbol(
		void *dst,
		const char *symbol, //! Either an addr of a var in the app, or a string
		size_t count, size_t offset,
		enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "dst=%p symb=%p, count=%lu\n", dst, symbol, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyDeviceToHost: id = CUDA_MEMCPY_FROM_SYMBOL_D2H; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_FROM_SYMBOL_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyFromSymbol(dst,symbol,count,offset,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpyFromSymbol(dst, symbol, count, offset, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyToArray(
		struct cudaArray *dst,
		size_t wOffset, size_t hOffset,
		const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "dst=%p wOffset=%lu, hOffset=%lu, src=%p, count=%lu\n",
			dst, wOffset, hOffset, src, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_TO_ARRAY_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_TO_ARRAY_D2D; break;
		default: BUG(1);
	}

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToArray(dst,wOffset,hOffset,src,count,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}


cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset,
		enum cudaMemcpyKind kind)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "symb=%p src=%p count=%lu\n", symbol, src, count);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_TO_SYMBOL_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_TO_SYMBOL_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToSymbol(symbol,src,count,offset,kind);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpyToSymbol(symbol, src, count, offset, kind, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemcpyToSymbolAsync(
		const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "symb %p\n", symbol);

    method_id_t id = CUDA_INVALID_METHOD;
	switch (kind) {
		case cudaMemcpyHostToDevice: id = CUDA_MEMCPY_H2D; break;
		case cudaMemcpyDeviceToDevice: id = CUDA_MEMCPY_D2D; break;
		default: BUG(1);
	}
#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemcpyToSymbolAsync(symbol,src,count,offset,kind,stream);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) + count;
#else
	cerr = assm_cudaMemcpyToSymbolAsync(symbol, src, count,
            offset, kind, stream, lat);
#endif

	update_latencies(lat, id);
	return cerr;
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemGetInfo(free,total);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaMemGetInfo(free, total, lat);
#endif

	printd(DBG_DEBUG, "free=%lu total=%lu\n", *free, *total);

	update_latencies(lat, CUDA_MEM_GET_INFO);
	return cerr;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
	cudaError_t cerr;
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaMemset(devPtr,value,count);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
	cerr = assm_cudaMemset(devPtr, value, count, lat);
#endif

	printd(DBG_DEBUG, "devPtr=%p value=%d count=%lu\n", devPtr, value, count);

	update_latencies(lat, CUDA_MEMSET);
	return cerr;
}


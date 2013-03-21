#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Texture Management
//===----------------------------------------------------------------------===//

// see comments in __cudaRegisterTexture and cudaBindTextureToArray
cudaError_t cudaBindTexture(size_t *offset,
		const struct textureReference *texRef, //! addr of global var in app
		const void *devPtr,
		const struct cudaChannelFormatDesc *desc,
		size_t size)
{
	cudaError_t cerr;
	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaBindTexture(offset,texRef,devPtr,desc,size);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_BIND_TEXTURE);
	return cerr;
}

cudaError_t cudaBindTextureToArray(
		const struct textureReference *texRef, //! address of global; copy full
		const struct cudaArray *array, //! use as pointer only
		const struct cudaChannelFormatDesc *desc) //! non-opaque; copied in full
{
	cudaError_t cerr;

	printd(DBG_DEBUG, "called\n");
    LAT_DECLARE(lat);
    trace_timestamp();

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	cerr = bypass.cudaBindTextureToArray(texRef,array,desc);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
    lat = NULL; /* compiler complains lat unused */
    abort(); /* XXX */
#endif

	update_latencies(lat, CUDA_BIND_TEXTURE_TO_ARRAY);
	return cerr;
}

struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
		enum cudaChannelFormatKind format)
{
#if 1
	// Doesn't need to be forwarded anywhere. Call the function in the NVIDIA
	// runtime, as this function just takes multiple variables and assigns them
	// to a struct. Why?
	const char *dl_err_str;
	typedef struct cudaChannelFormatDesc (*func)
		(int,int,int,int,enum cudaChannelFormatKind);
	func f = (func)dlsym(RTLD_NEXT, __func__);
	dl_err_str = dlerror();
	if (dl_err_str) {
		fprintf(stderr, USERMSG_PREFIX
				" error: could not find NVIDIA runtime: %s\n", dl_err_str);
		BUG(0);
	}
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u;"
			" passing to NV runtime\n",
			x, y, z, w, format);
	return f(x,y,z,w,format);
#else
	int err;

	struct cuda_packet *shmpkt;
	printd(DBG_DEBUG, "x=%d y=%d z=%d w=%d format=%u\n",
			x, y, z, w, format);

	TIMER_DECLARE1(t);
	TIMER_START(t);
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = CUDA_CREATE_CHANNEL_DESC;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].arr_argii[0] = x;
	shmpkt->args[0].arr_argii[1] = y;
	shmpkt->args[0].arr_argii[2] = z;
	shmpkt->args[0].arr_argii[3] = w;
	shmpkt->args[1].arr_arguii[0] = format;
	shmpkt->len = sizeof(*shmpkt);
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);

	update_latencies(&shmpkt->lat);
	return shmpkt->args[0].desc;
#endif
}

#if 0
cudaError_t cudaGetTextureReference(
		const struct textureReference **texRef,
		const char *symbol)
{
	struct cuda_packet *shmpkt =
		(struct cuda_packet *)get_region(pthread_self());

	printd(DBG_DEBUG, "symbol=%p\n", symbol);

	BUG(0);

	return shmpkt->ret_ex_val.err;
}
#endif


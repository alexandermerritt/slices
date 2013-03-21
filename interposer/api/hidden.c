#include "../lib.h"

//===----------------------------------------------------------------------===//
// CUDA Runtime API - Hidden Registration
//===----------------------------------------------------------------------===//

void** __cudaRegisterFatBinary(void* cubin)
{
	void **handle;
	//struct cuda_packet *shmpkt;
    LAT_DECLARE(lat);
    trace_timestamp();

	if (num_registered_cubins <= 0) {
        fill_bypass(&bypass);
	}

	num_registered_cubins++;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	handle = bypass.__cudaRegisterFatBinary(cubin);
	TIMER_END(t, lat->exec.call);
	cache_num_entries_t _notused;
	lat->len = sizeof(struct cuda_packet) + getFatRecPktSize(cubin, &_notused);
#else
    handle = assm__cudaRegisterFatBinary(cubin, lat);
	printd(DBG_DEBUG, "handle=%p\n", handle);
#endif

	update_latencies(lat, __CUDA_REGISTER_FAT_BINARY);
	return handle;
}

void __cudaUnregisterFatBinary(void** fatCubinHandle)
{
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_INFO, "handle=%p\n", fatCubinHandle);

	num_registered_cubins--;

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaUnregisterFatBinary(fatCubinHandle);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet);
#else
    assm__cudaUnregisterFatBinary(fatCubinHandle, lat);
#endif

	update_latencies(lat, __CUDA_UNREGISTER_FAT_BINARY);

	if (num_registered_cubins <= 0) { // only detach on last unregister
		print_latencies();
	}

	return;
}

void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize)
{
    LAT_DECLARE(lat);
    trace_timestamp();

	printd(DBG_DEBUG, "handle=%p hostFun=%p deviceFun=%s deviceName=%s\n",
			fatCubinHandle, hostFun, deviceFun, deviceName);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaRegisterFunction(fatCubinHandle,hostFun,deviceFun,deviceName,
			thread_limit,tid,bid,bDim,gDim,wSize);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) +
		getSize_regFuncArgs(fatCubinHandle, hostFun, deviceFun, deviceName,
				thread_limit, tid, bid, bDim, gDim, wSize);
#else
    assm__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
            deviceName, thread_limit, tid, bid, bDim, gDim, wSize, lat);
#endif

	update_latencies(lat, __CUDA_REGISTER_FUNCTION);
	return;
}

void __cudaRegisterVar(
		void **fatCubinHandle,	//! cubin this var associates with
		char *hostVar,			//! addr of a var within app (not string)
		char *deviceAddress,	//! 8-byte device addr
		const char *deviceName, //! actual string
		int ext, int vsize, int constant, int global)
{
    LAT_DECLARE(lat);
    trace_timestamp();
	printd(DBG_DEBUG, "symbol=%p\n", hostVar);

#if defined(TIMING) && defined(TIMING_NATIVE)
	TIMER_DECLARE1(t);
	TIMER_START(t);
	bypass.__cudaRegisterVar(fatCubinHandle,hostVar,deviceAddress,deviceName,
			ext,vsize,constant,global);
	TIMER_END(t, lat->exec.call);
	lat->len = sizeof(struct cuda_packet) +
		getSize_regVar(fatCubinHandle, hostVar, deviceAddress, deviceName,
				ext, vsize, constant, global);
#else
	assm__cudaRegisterVar(fatCubinHandle,hostVar,deviceAddress,deviceName,
			ext,vsize,constant,global, lat);
#endif

	update_latencies(lat, __CUDA_REGISTER_VARIABLE);
	return;
}

#if 0
// first three args we treat as handles (i.e. only pass down the pointer
// addresses)
void __cudaRegisterTexture(
		void** fatCubinHandle,
		//! address of a global variable within the application; store the addr
		const struct textureReference* texRef,
		//! 8-byte device address; dereference it once to get the address
		const void** deviceAddress,
		const char* texName, //! actual string
		int dim, int norm, int ext)
{
	struct cuda_packet *shmpkt;
	TIMER_DECLARE1(t);

	printd(DBG_DEBUG, "handle=%p texRef=%p devAddr=%p *devAddr=%p texName=%s"
			" dim=%d norm=%d ext=%d\n",
			fatCubinHandle, texRef, deviceAddress, *deviceAddress, texName,
			dim, norm, ext);

	TIMER_START(t);
#if defined(TIMING) && defined(TIMING_NATIVE)
	struct cuda_packet tpkt; // XXX this might add a lot to the stack
	memset(&tpkt, 0, sizeof(tpkt));
	bypass.__cudaRegisterTexture(fatCubinHandle,texRef,deviceAddress,texName,
			dim,norm,ext);
	TIMER_END(t, tpkt.lat.exec.call);
	tpkt.len = sizeof(tpkt) + strlen(texName) + 1;
	shmpkt = &tpkt;
#else
	uintptr_t shm_ptr;
	shmpkt = (struct cuda_packet *)get_region(pthread_self());
	shm_ptr = (uintptr_t)shmpkt;
	memset(shmpkt, 0, sizeof(*shmpkt));
	shmpkt->method_id = __CUDA_REGISTER_TEXTURE;
	shmpkt->thr_id = pthread_self();
	shmpkt->args[0].argdp = fatCubinHandle; // pointer copy
	// Subsequent calls to texture functions will provide the address of this
	// global as a parameter; there we will copy the data in the global into the
	// cuda_packet, copy it to the variable registered in the sink process, and
	// invoke such functions with the variable address allocated in the sink. We
	// still need this address, because we cannot tell the application to change
	// which address it provides us---we have to perform lookups of the global's
	// address in the application with what we cache in the sink process.
	shmpkt->args[1].argp = (void*)texRef; // pointer copy
	shmpkt->args[2].texRef = *texRef; // state copy
	shmpkt->args[3].argp = (void*)*deviceAddress; // copy actual address
	shmpkt->args[4].argull = sizeof(*shmpkt); // offset
	shm_ptr += sizeof(*shmpkt);
	memcpy((void*)shm_ptr, texName, strlen(texName) + 1);
	shmpkt->args[5].arr_argii[0] = dim;
	shmpkt->args[5].arr_argii[1] = norm;
	shmpkt->args[5].arr_argii[2] = ext;
	shmpkt->len = sizeof(*shmpkt) + strlen(texName) + 1;
	shmpkt->is_sync = true;
	TIMER_END(t, shmpkt->lat.lib.setup);

	TIMER_START(t);
	//assembly_rpc(assm_id, 0, shmpkt);
	TIMER_END(t, shmpkt->lat.lib.wait);
#endif

	update_latencies(lat, __CUDA_REGISTER_TEXTURE);
	return;
}
#endif

/**
 * @file local_api_wrapper.h
 * @brief copied from remote_gpu/nvidia_backend/local_api_wrapper.h
 *
 * @date Mar 3, 2011
 * @author Adapted by Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#ifndef LOCAL_API_WRAPPER_H_
#define LOCAL_API_WRAPPER_H_

// the original function we eventually want to invoked
extern void** __cudaRegisterFatBinary(void* fatC);
extern void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);

int __nvback_cudaRegisterFatBinary(cuda_packet_t *packet);

void *copyFatBinary(__cudaFatCudaBinary *fatCubin);

#endif /* LOCAL_API_WRAPPER_H_ */

/**
 * @file cuda_hidden.h
 * @author Alex Merritt
 * @date 2011-11-13
 * @brief This header contains prototypes for undocumented functions within the
 * proprietary CUDA Runtime API that are needed for interposing.
 */

#ifndef _CUDA_HIDDEN_H
#define _CUDA_HIDDEN_H

extern void** __cudaRegisterFatBinary(void*);

extern void __cudaUnregisterFatBinary(void**);

extern void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);

extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global);

// TODO register texture

#endif

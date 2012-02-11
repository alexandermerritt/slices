/**
 * @file cuda_hidden.h
 * @author Alex Merritt
 * @date 2011-11-13
 * @brief This header contains prototypes for undocumented functions within the
 * proprietary CUDA Runtime API that are needed for interposing.
 */

#ifndef _CUDA_HIDDEN_H
#define _CUDA_HIDDEN_H

#include <cuda_runtime_api.h>

extern void** __cudaRegisterFatBinary(void*);

extern void __cudaUnregisterFatBinary(void**);

extern void __cudaRegisterFunction(
		void** fatCubinHandle,		//! Pointer to (heap?) memory
		const char* hostFun,		//! Symbol (function) pointer
		char* deviceFun,			//! String
		const char* deviceName,		//! String
		int thread_limit, uint3* tid, uint3* bid,
		dim3* bDim, dim3* gDim, int* wSize);

/**
 * Andrew Kerr: "this function establishes a mapping between global variables
 * defined in .ptx or .cu modules and host-side variables. In PTX, global
 * variables have module scope and can be globally referenced by module and
 * variable name. In the CUDA Runtime API, globals in two modules must not have
 * the same name."
 *
 * This function is generated when you declare a variable in global or constant
 * memory on the GPU within your code.
 */
extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize,
		int constant, int global);

/**
 * This function is invoked when you declare a texture variable as a global
 * within your code.
 */
extern void __cudaRegisterTexture(void** fatCubinHandle, const struct
		textureReference* hostVar, const void** deviceAddress, const char
		*texName, int dim, int norm, int ext);

// TODO register texture

#endif

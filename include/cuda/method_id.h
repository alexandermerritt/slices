/**
 * @file method_id.h
 * @brief Contains debug utils
 * Copied from: remote_gpu/include/method_id.h
 *
 * @date Feb 28, 2011
 * @author Prepared by Magda Slawinska, magg@gatech.edu
 */

#ifndef __METHOD_ID_H
#define __METHOD_ID_H

// Method IDs we assign to API functions.
typedef enum METHOD_ID {
	CUDA_INVALID_METHOD = 0,
    CUDA_MALLOC = 1, // don't set this to zero to catch indicies set to zero then not modified
	CUDA_HOST_ALLOC,
	CUDA_MALLOC_PITCH,
	CUDA_MALLOC_ARRAY,
    CUDA_FREE,
    CUDA_FREE_ARRAY,
    CUDA_MEMCPY_H2D,
    CUDA_MEMCPY_D2H,
    CUDA_MEMCPY_H2H,
    CUDA_MEMCPY_D2D,
    CUDA_MEMCPY_TO_ARRAY_D2D,
    CUDA_MEMCPY_TO_ARRAY_H2D,
	CUDA_MEMCPY_ASYNC_H2D,
	CUDA_MEMCPY_ASYNC_D2H,
	CUDA_MEMCPY_ASYNC_H2H,
	CUDA_MEMCPY_ASYNC_D2D,
    CUDA_SETUP_ARGUMENT,
    CUDA_LAUNCH,
    CUDA_GET_DEVICE_COUNT,
    CUDA_GET_DEVICE_PROPERTIES,
    CUDA_GET_DEVICE,
    CUDA_SET_DEVICE,
    CUDA_SET_DEVICE_FLAGS,
    CUDA_SET_VALID_DEVICES,
    CUDA_CONFIGURE_CALL,
    CUDA_THREAD_SYNCHRONIZE,
    CUDA_THREAD_EXIT,
    CUDA_MEMSET,
    CUDA_UNBIND_TEXTURE,
    CUDA_BIND_TEXTURE_TO_ARRAY,
    CUDA_FREE_HOST,
    CUDA_MEMCPY_TO_SYMBOL_H2D,
    CUDA_MEMCPY_TO_SYMBOL_D2D,
    CUDA_MEMCPY_FROM_SYMBOL_D2H,
    CUDA_MEMCPY_FROM_SYMBOL_D2D,
    CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D,
    CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D,
    CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2H,
    CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2D,
    CUDA_MEMCPY_2D_TO_ARRAY_D2D,
    CUDA_MEMCPY_2D_TO_ARRAY_H2D,
    CUDA_MEMCPY_2D_TO_ARRAY_D2H,
    CUDA_MEMCPY_2D_TO_ARRAY_H2H,
	CUDA_MEM_GET_INFO,
    __CUDA_REGISTER_FAT_BINARY,
    __CUDA_REGISTER_FUNCTION,
    __CUDA_REGISTER_VARIABLE,
    __CUDA_REGISTER_TEXTURE,
    __CUDA_REGISTER_SHARED,
    __CUDA_UNREGISTER_FAT_BINARY,
	CUDA_DRIVER_GET_VERSION,
	CUDA_RUNTIME_GET_VERSION,
	CUDA_FUNC_GET_ATTR,
	CUDA_BIND_TEXTURE,
	CUDA_GET_TEXTURE_REFERENCE,
	CUDA_STREAM_CREATE,
	CUDA_STREAM_SYNCHRONIZE,
	CUDA_CREATE_CHANNEL_DESC,
	CUDA_METHOD_LIMIT
} method_id_t;

#endif

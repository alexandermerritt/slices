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
    CUDA_MEMCPY_H2H,
    CUDA_MEMCPY_H2D,
    CUDA_MEMCPY_D2H,
    CUDA_MEMCPY_D2D,
    CUDA_MEMCPY_2D_H2H,
    CUDA_MEMCPY_2D_H2D,
    CUDA_MEMCPY_2D_D2H,
    CUDA_MEMCPY_2D_D2D,
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
	CUDA_STREAM_DESTROY,
	CUDA_STREAM_QUERY,
	CUDA_STREAM_SYNCHRONIZE,
	CUDA_CREATE_CHANNEL_DESC,
	CUDA_EVENT_CREATE,
	CUDA_EVENT_CREATE_WITH_FLAGS,
	CUDA_EVENT_RECORD,
	CUDA_EVENT_QUERY,
	CUDA_EVENT_SYNCHRONIZE,
	CUDA_EVENT_DESTROY,
	CUDA_EVENT_ELAPSED_TIME,
	CUDA_METHOD_LIMIT
} method_id_t;

/** structure indicating whether a func is to be handled synchronously */
static const bool
method_synctable[CUDA_METHOD_LIMIT] = {
	[CUDA_INVALID_METHOD]                =  false,  //  not a real func
	[CUDA_MALLOC]                        =  true,
	[CUDA_HOST_ALLOC]                    =  true,
	[CUDA_MALLOC_PITCH]                  =  true,
	[CUDA_MALLOC_ARRAY]                  =  true,
	[CUDA_FREE]                          =  false,
	[CUDA_FREE_ARRAY]                    =  false,
	[CUDA_MEMCPY_H2H]                    =  true,
	[CUDA_MEMCPY_H2D]                    =  false,
	[CUDA_MEMCPY_D2H]                    =  true,
	[CUDA_MEMCPY_D2D]                    =  true,   //  reexamine
	[CUDA_MEMCPY_2D_H2H]                 =  true,
	[CUDA_MEMCPY_2D_H2D]                 =  false,  //  reexamine
	[CUDA_MEMCPY_2D_D2H]                 =  true,   //  reexamine
	[CUDA_MEMCPY_2D_D2D]                 =  false,  //  reexamine
	[CUDA_MEMCPY_TO_ARRAY_D2D]           =  true,   //  reexamine
	[CUDA_MEMCPY_TO_ARRAY_H2D]           =  false,
	[CUDA_MEMCPY_ASYNC_H2D]              =  false,
	[CUDA_MEMCPY_ASYNC_D2H]              =  true,
	[CUDA_MEMCPY_ASYNC_H2H]              =  true,
	[CUDA_MEMCPY_ASYNC_D2D]              =  true,   //  reexamine
	[CUDA_SETUP_ARGUMENT]                =  false,
	[CUDA_LAUNCH]                        =  false,
	[CUDA_GET_DEVICE_COUNT]              =  true,
	[CUDA_GET_DEVICE_PROPERTIES]         =  true,
	[CUDA_GET_DEVICE]                    =  true,
	[CUDA_SET_DEVICE]                    =  false,
	[CUDA_SET_DEVICE_FLAGS]              =  false,
	[CUDA_SET_VALID_DEVICES]             =  false,
	[CUDA_CONFIGURE_CALL]                =  false,
	[CUDA_THREAD_SYNCHRONIZE]            =  true,
	[CUDA_THREAD_EXIT]                   =  true,
	[CUDA_MEMSET]                        =  false,
	[CUDA_UNBIND_TEXTURE]                =  false,
	[CUDA_BIND_TEXTURE_TO_ARRAY]         =  false,
	[CUDA_FREE_HOST]                     =  false,
	[CUDA_MEMCPY_TO_SYMBOL_H2D]          =  false,
	[CUDA_MEMCPY_TO_SYMBOL_D2D]          =  true,   //  reexamine
	[CUDA_MEMCPY_FROM_SYMBOL_D2H]        =  true,
	[CUDA_MEMCPY_FROM_SYMBOL_D2D]        =  true,   //  reexamine
	[CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D]    =  false,
	[CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D]    =  true,   //  reexamine
	[CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2H]  =  true,
	[CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2D]  =  true,   //  reexamine
	[CUDA_MEMCPY_2D_TO_ARRAY_D2D]        =  true,   //  reexamine
	[CUDA_MEMCPY_2D_TO_ARRAY_H2D]        =  false,
	[CUDA_MEMCPY_2D_TO_ARRAY_D2H]        =  true,
	[CUDA_MEMCPY_2D_TO_ARRAY_H2H]        =  true,
	[CUDA_MEM_GET_INFO]                  =  true,
	[__CUDA_REGISTER_FAT_BINARY]         =  true,
	[__CUDA_REGISTER_FUNCTION]           =  false,
	[__CUDA_REGISTER_VARIABLE]           =  false,
	[__CUDA_REGISTER_TEXTURE]            =  false,
	[__CUDA_REGISTER_SHARED]             =  false,
	[__CUDA_UNREGISTER_FAT_BINARY]       =  true,
	[CUDA_DRIVER_GET_VERSION]            =  true,
	[CUDA_RUNTIME_GET_VERSION]           =  true,
	[CUDA_FUNC_GET_ATTR]                 =  true,
	[CUDA_BIND_TEXTURE]                  =  false,
	[CUDA_GET_TEXTURE_REFERENCE]         =  true,
	[CUDA_STREAM_CREATE]                 =  true,
	[CUDA_STREAM_DESTROY]                =  false,
	[CUDA_STREAM_QUERY]                  =  false,
	[CUDA_STREAM_SYNCHRONIZE]            =  true,
	[CUDA_CREATE_CHANNEL_DESC]           =  true,
	[CUDA_EVENT_CREATE]                  =  true,
	[CUDA_EVENT_CREATE_WITH_FLAGS]       =  true,
	[CUDA_EVENT_RECORD]                  =  false,
	[CUDA_EVENT_QUERY]                   =  true,
	[CUDA_EVENT_SYNCHRONIZE]             =  true,
	[CUDA_EVENT_DESTROY]                 =  false,
	[CUDA_EVENT_ELAPSED_TIME]            =  true
};

static const char *
method2str(method_id_t id)
{
	switch (id) {
		case CUDA_INVALID_METHOD: return "CUDA_INVALID_METHOD";
		case CUDA_MALLOC: return "CUDA_MALLOC";
		case CUDA_HOST_ALLOC: return "CUDA_HOST_ALLOC";
		case CUDA_MALLOC_PITCH: return "CUDA_MALLOC_PITCH";
		case CUDA_MALLOC_ARRAY: return "CUDA_MALLOC_ARRAY";
		case CUDA_FREE: return "CUDA_FREE";
		case CUDA_FREE_ARRAY: return "CUDA_FREE_ARRAY";
		case CUDA_MEMCPY_H2H: return "CUDA_MEMCPY_H2H";
		case CUDA_MEMCPY_H2D: return "CUDA_MEMCPY_H2D";
		case CUDA_MEMCPY_D2H: return "CUDA_MEMCPY_D2H";
		case CUDA_MEMCPY_D2D: return "CUDA_MEMCPY_D2D";
		case CUDA_MEMCPY_TO_ARRAY_D2D: return "CUDA_MEMCPY_TO_ARRAY_D2D";
		case CUDA_MEMCPY_TO_ARRAY_H2D: return "CUDA_MEMCPY_TO_ARRAY_H2D";
		case CUDA_MEMCPY_ASYNC_H2D: return "CUDA_MEMCPY_ASYNC_H2D";
		case CUDA_MEMCPY_ASYNC_D2H: return "CUDA_MEMCPY_ASYNC_D2H";
		case CUDA_MEMCPY_ASYNC_H2H: return "CUDA_MEMCPY_ASYNC_H2H";
		case CUDA_MEMCPY_ASYNC_D2D: return "CUDA_MEMCPY_ASYNC_D2D";
		case CUDA_SETUP_ARGUMENT: return "CUDA_SETUP_ARGUMENT";
		case CUDA_LAUNCH: return "CUDA_LAUNCH";
		case CUDA_GET_DEVICE_COUNT: return "CUDA_GET_DEVICE_COUNT";
		case CUDA_GET_DEVICE_PROPERTIES: return "CUDA_GET_DEVICE_PROPERTIES";
		case CUDA_GET_DEVICE: return "CUDA_GET_DEVICE";
		case CUDA_SET_DEVICE: return "CUDA_SET_DEVICE";
		case CUDA_SET_DEVICE_FLAGS: return "CUDA_SET_DEVICE_FLAGS";
		case CUDA_SET_VALID_DEVICES: return "CUDA_SET_VALID_DEVICES";
		case CUDA_CONFIGURE_CALL: return "CUDA_CONFIGURE_CALL";
		case CUDA_THREAD_SYNCHRONIZE: return "CUDA_THREAD_SYNCHRONIZE";
		case CUDA_THREAD_EXIT: return "CUDA_THREAD_EXIT";
		case CUDA_MEMSET: return "CUDA_MEMSET";
		case CUDA_UNBIND_TEXTURE: return "CUDA_UNBIND_TEXTURE";
		case CUDA_BIND_TEXTURE_TO_ARRAY: return "CUDA_BIND_TEXTURE_TO_ARRAY";
		case CUDA_FREE_HOST: return "CUDA_FREE_HOST";
		case CUDA_MEMCPY_TO_SYMBOL_H2D: return "CUDA_MEMCPY_TO_SYMBOL_H2D";
		case CUDA_MEMCPY_TO_SYMBOL_D2D: return "CUDA_MEMCPY_TO_SYMBOL_D2D";
		case CUDA_MEMCPY_FROM_SYMBOL_D2H: return "CUDA_MEMCPY_FROM_SYMBOL_D2H";
		case CUDA_MEMCPY_FROM_SYMBOL_D2D: return "CUDA_MEMCPY_FROM_SYMBOL_D2D";
		case CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D: return "CUDA_MEMCPY_TO_SYMBOL_ASYNC_H2D";
		case CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D: return "CUDA_MEMCPY_TO_SYMBOL_ASYNC_D2D";
		case CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2H: return "CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2H";
		case CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2D: return "CUDA_MEMCPY_FROM_SYMBOL_ASYNC_D2D";
		case CUDA_MEMCPY_2D_TO_ARRAY_D2D: return "CUDA_MEMCPY_2D_TO_ARRAY_D2D";
		case CUDA_MEMCPY_2D_TO_ARRAY_H2D: return "CUDA_MEMCPY_2D_TO_ARRAY_H2D";
		case CUDA_MEMCPY_2D_TO_ARRAY_D2H: return "CUDA_MEMCPY_2D_TO_ARRAY_D2H";
		case CUDA_MEMCPY_2D_TO_ARRAY_H2H: return "CUDA_MEMCPY_2D_TO_ARRAY_H2H";
		case CUDA_MEM_GET_INFO: return "CUDA_MEM_GET_INFO";
		case __CUDA_REGISTER_FAT_BINARY: return "__CUDA_REGISTER_FAT_BINARY";
		case __CUDA_REGISTER_FUNCTION: return "__CUDA_REGISTER_FUNCTION";
		case __CUDA_REGISTER_VARIABLE: return "__CUDA_REGISTER_VARIABLE";
		case __CUDA_REGISTER_TEXTURE: return "__CUDA_REGISTER_TEXTURE";
		case __CUDA_REGISTER_SHARED: return "__CUDA_REGISTER_SHARED";
		case __CUDA_UNREGISTER_FAT_BINARY: return "__CUDA_UNREGISTER_FAT_BINARY";
		case CUDA_DRIVER_GET_VERSION: return "CUDA_DRIVER_GET_VERSION";
		case CUDA_RUNTIME_GET_VERSION: return "CUDA_RUNTIME_GET_VERSION";
		case CUDA_FUNC_GET_ATTR: return "CUDA_FUNC_GET_ATTR";
		case CUDA_BIND_TEXTURE: return "CUDA_BIND_TEXTURE";
		case CUDA_GET_TEXTURE_REFERENCE: return "CUDA_GET_TEXTURE_REFERENCE";
		case CUDA_STREAM_CREATE: return "CUDA_STREAM_CREATE";
		case CUDA_STREAM_DESTROY: return "CUDA_STREAM_DESTROY";
		case CUDA_STREAM_QUERY: return "CUDA_STREAM_QUERY";
		case CUDA_STREAM_SYNCHRONIZE: return "CUDA_STREAM_SYNCHRONIZE";
		case CUDA_CREATE_CHANNEL_DESC: return "CUDA_CREATE_CHANNEL_DESC";
		case CUDA_METHOD_LIMIT: return "CUDA_METHOD_LIMIT";
		default: return "INVALID_METHOD_ID";
	}
}

#endif

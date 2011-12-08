/**
 * @file packetheader.h
 * @brief copied from remote_gpu/include/packetheader.h and modified
 *
 * @date Feb 23, 2011
 * @author Magda Slawinska, magg@gatech.edu
 *
 * @date 2011-11-04
 * @author Alex Merritt, merritt.alex@gatech.edu
 * Updated registration structures (tex, var, func) to contain list variables
 * for linking together.
 */

#ifndef PACKETHEADER_H_
#define PACKETHEADER_H_

// System includes
#include <pthread.h>
#include <stdint.h>

// CUDA includes
#include <driver_types.h>
#include <vector_types.h>

// Project includes
#include <method_id.h>
#include <util/list.h>

typedef uint32_t grant_ref_t;
typedef pthread_t tid_t;

#define MAX_ARGS 5

enum cuda_packet_flags
{
	CUDA_PKT_REQUEST = 0x1,		/* This packet is flowing from app to assembly. */
	CUDA_PKT_RESPONSE  = 0x2,	/* Assembly -> app; RPC already executed somewhere. */
	//CUDA_PKT_MORE_DATA = 0x4, /* Not used. */
	//CUDA_PKT_ERROR    = 0x8		/* Used to indicate if the RPC produced an error. */
	//CUDA_PKT_PTR_DATA = 0x10, /* Not used. */
	//CUDA_PKT_ADDR_MAPPED = 0x20, /* Not used. */
	//CUDA_PKT_MEM_SHARED = 0x40, /* Not used. */
};
// TODO Create macros to access and modify the value of 'flags' in a cuda packet
// instead of having code manually do raw bit ops everywhere.

// XXX Move reg_*_args_t to fatcubininfo.h
// And name that file something else
// and move that file to include/cuda/

// The cudaRegisterFunction() has 10 arguments. Instead of passing them in
// multiple packet rounds, it is serialized onto as many pages and can be
// accessed by using this struct
//
// Refer to cuda_hidden.h for the meanings of these variables.
typedef struct {
	struct list_head link;
	void** fatCubinHandle;
	char* hostFun;
    char* deviceFun;
	char* deviceName;
	int thread_limit;
	uint3* tid;
    uint3* bid;
	dim3* bDim;
	dim3* gDim;
	int* wSize;
} reg_func_args_t;

// The __cudaRegisterVar() has 8 arguments. Pass them in one page
typedef struct {
	struct list_head link;
	void **fatCubinHandle;
	char *hostVar;  // Address coming from Guest thru RegisterVar
	char *dom0HostAddr;  // This addr will be registered with cuda driver instead
	char *deviceAddress;
	char *deviceName;
	int ext;
	int size;
	int constant;
	int global;
} reg_var_args_t;

// The __cudaRegisterTex() has 8 arguments. Pass them in one page
typedef struct {
	struct list_head link;
	void **fatCubinHandle;
	struct textureReference *texref; //! address of global in application
	struct textureReference tex; //! actual storage registered within sink
	void **devPtr;
	char *devName;
	int dim;
	int norm;
	int ext;
} reg_tex_args_t;

// Arguments to functions can either be a simple argument or grant reference
// Left to the function to decipher
// Arguments will be filled in the order as in function declaration
// Currently this union has the possible arguments seen in common CUDA calls
typedef union args {
	int arr_argii[4];
	unsigned int arr_arguii[4];
	long long argll;
	unsigned long long argull;          // for pointers and such
	float argf;
	void *argp;
	void **argdp;
	char *argcp;
	size_t arr_argi[2];
	unsigned long long arr_argui[2];
	dim3 arg_dim;                       // 3D point
	cudaStream_t stream;
	struct cudaArray *cudaArray; // this is an opaque type; contains a handle
	struct cudaChannelFormatDesc desc;
	struct textureReference texref;
} args_t;

// Possible return types in a response or some extra information on the way to
// the backend
typedef union ret_extra {
	int num_args;       // tells backend the number of args in case of map_pages_in_backend
	int bit_idx;        // Pass idx into bitmap for mmap cases (HACK)
	uint32_t data_unit; // tells how to interpret size
	cudaError_t err;    // most common return type
	cudaError_t *errp;  // return type for cudaMalloc so far
	const char *charp;  // seen this one somewhere
	void **handle;	    // Used to return fatCubinHandle
} ret_extra_t;

typedef struct cuda_packet {
	method_id_t method_id;     // to identify which method
	uint16_t req_id;        // to identify which request is the response for in case async
	tid_t thr_id;           // thread sending request
	uint8_t flags;          // if ever needed to indicate more data for the same call
	args_t args[MAX_ARGS];  // arguments to be copied on ring
	ret_extra_t ret_ex_val; // return value from call filled in response packet
							// when coming from backend or extra information from
							// frontend
} cuda_packet_t;


#endif /* PACKETHEADER_H_ */

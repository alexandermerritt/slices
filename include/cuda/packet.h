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
#include <stdbool.h>
#include <stdint.h>

// CUDA includes
#include <driver_types.h>
#include <vector_types.h>

// Project includes
#include <cuda/method_id.h>
#include <util/list.h>

#define MAX_ARGS 6

// forward declaration
struct cuda_packet;

// TODO Create macros to access and modify the value of 'flags' in a cuda packet
// instead of having code manually do raw bit ops everywhere.
enum cuda_packet_flags
{
	/** RPC newly created from interposer. Direction of packet is towards
	 * assembly */
	CUDA_PKT_REQUEST = 0x1,
	/** RPC was executed somewhere. This packet is on the return path to the
	 * application */
	CUDA_PKT_RESPONSE  = 0x2,
	/** RPC is a function that uses a symbol that was provided by the caller as
	 * a string, instead of an address (the default is to assume a symbol is an
	 * address): symbols can either be the address of a variable within the
	 * application address space, or the address of a string literal containing
	 * the name of said variable. Enabling this flag tells the runtime that the
	 * packet argument contains an offset into the memory region where the
	 * string has been copied. If not set, the packet argument contains the
	 * memory address of the variable directly. */
	CUDA_PKT_SYMB_IS_STRING = 0x4,
};

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
	// FIXME Rename this variable
	char *hostVar;  // Address coming from Guest thru RegisterVar
	// FIXME Rename this variable
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
	struct textureReference *texRef; //! address of global in application
	struct textureReference tex; //! actual storage registered within sink
	const void *devPtr;
	const char *texName;
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
	cudaEvent_t event;
	cudaStream_t stream;
	struct cudaArray *cudaArray; // this is an opaque type; contains a handle
	struct cudaChannelFormatDesc desc;
	struct textureReference texRef;
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
	pthread_t thr_id;           // thread sending request
	uint8_t flags;          // if ever needed to indicate more data for the same call
	args_t args[MAX_ARGS];  // arguments to be copied on ring
	size_t len; //! total bytes of marshalled packet incl appended data
	bool is_sync; //! whether this func must be interposed/invoked synchronously
	ret_extra_t ret_ex_val; // return value from call filled in response packet
} cuda_packet_t;

/* cudaError_t return value of RPC packet */
static inline cudaError_t
cpkt_ret_err(void *buf)
{
    return ((struct cuda_packet*)(buf))->ret_ex_val.err;
}
static inline void**
cpkt_ret_hdl(void *buf)
{
    return ((struct cuda_packet*)(buf))->ret_ex_val.handle;
}

//! Data type describing an offset into the batch buffer. Cannot realistically
//! be smaller than uint if the buffer size is anything reasonable.
typedef unsigned int offset_t;

//! Absolute maximum number of packets a batch can hold, as the offset array is
//! allocated at compile time and is included in the batch header.
#define CUDA_BATCH_MAX			8192 /* XXX hardcoded ... */
#define CUDA_BATCH_BUFFER_SZ	((1UL << 30) + (256UL << 20)) /* XXX hardcoded ... */

/** size at which SDP switches to ZCopy */
#define ZCPY_TRIGGER_SZ			(64 << 10)

/** additional bytes needed in the header to trigger ZCopy */
#define PADDING_DIFF			(ZCPY_TRIGGER_SZ - (sizeof(offset_t) * CUDA_BATCH_MAX))
#define PADDING_BYTES			(ZCPY_TRIGGER_SZ < (sizeof(offset_t) * CUDA_BATCH_MAX) ? 0 : (PADDING_DIFF))

struct cuda_pkt_batch {
	struct {
		size_t num_pkts; //! offsets used
		size_t bytes_used; //! in the buffer
		/** Offsets of packets within buffer. Offsets specified within packet
		 * arguments are relative to the address of the packet itself.  XXX Be
		 * careful the storage type of 'offsets' is able to hold offset values
		 * into the storage pointed to by 'buffer'. */
		offset_t offsets[CUDA_BATCH_MAX];
#if defined(NIC_SDP)
		unsigned char _SDPZCopyPadding[PADDING_BYTES];
#endif
	} header;

	size_t max; //! Maximum number of packets allowed
	void *buffer;
};

#endif /* PACKETHEADER_H_ */

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
#include <util/timer.h>

#define MAX_ARGS 6

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

#ifdef TIMING
/**
 * Measurements of time spent by a cuda_packet RPC within each component of the
 * runtime. Only allocated/updated if macro TIMING is defined.
 */
struct rpc_latencies {
	// If you measure the total time of the application, then subtract from it
	// the attach and detach latencies (joining/departing the runtime) as well
	// as lib.setup and lib.wait, you end up approx. with the time in the
	// application spent NOT using CUDA.
	struct {
		uint64_t setup; //! Time spent marshaling and misc setup
		//! Time spent polling for result. Composed of all costs in the assembly
		//! runtime executing the call, locally or remote.
		uint64_t wait;
	} lib; // interposer overhead
	struct {
		uint64_t setup; //! Argument setup and symbol/cubin lookup time
		uint64_t call; //! Latency in the CUDA runtime/driver
	} exec; // cuda/execute.c either local- or remote-tip execution
	struct {
		uint64_t append; //! Time squandered doing memcpy to the batch buffer
		uint64_t send; //! Time spent sending the batch
		//! Time spent waiting for the return packet (and receipt of said
		//! packet). Includes time spent at remote machine executing RPCs
		//! (exec.setup + exec.call).
		uint64_t wait;
		uint64_t recv; //! Time spent receiving return data (if required)
	} rpc; // cuda/rpc.c (all zeros if vgpu is not remote)
	struct {
		// rpc.{send|wait|recv} - batch_exec = time on network
		// batch_exec - exec.{setup|call} = batch unpacking
		uint64_t batch_exec; //! Executing all RPCs in a batch
	} remote; // on remote machine (all zeros if vgpu is not remote)
};
#endif	/* TIMING */

typedef struct cuda_packet {
	method_id_t method_id;     // to identify which method
	uint16_t req_id;        // to identify which request is the response for in case async
	pthread_t thr_id;           // thread sending request
	uint8_t flags;          // if ever needed to indicate more data for the same call
	args_t args[MAX_ARGS];  // arguments to be copied on ring
	size_t len; //! total bytes of marshalled packet incl appended data
	bool is_sync; //! whether this func must be interposed/invoked synchronously
	ret_extra_t ret_ex_val; // return value from call filled in response packet
#ifdef TIMING
	struct rpc_latencies lat;
#endif
} cuda_packet_t;

//! Absolute maximum number of packets a batch can hold, as the offset array is
//! allocated at compile time and is included in the batch header.
#define CUDA_BATCH_MAX			16384UL
#define CUDA_BATCH_BUFFER_SZ	(512 << 20)

struct cuda_pkt_batch {
	struct {
		size_t num_pkts; //! packets stored in this batch; limited by CUDA_BATCH_MAX
		//! Offsets of packets within buffer. Offsets specified within packet
		//! arguments are relative to the address of the packet itself.
		// XXX Be careful the storage type of 'offsets' is able to hold offset
		// values into the storage pointed to by 'buffer'.
		unsigned int offsets[CUDA_BATCH_MAX];
		size_t bytes_used;
	} header;
	size_t max; //! Maximum number of packets allowed
	void *buffer;
};

#endif /* PACKETHEADER_H_ */

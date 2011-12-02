/**
 * @file common/cuda/rpc.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief Send serialized CUDA packets across the network for another machine to
 * execute. Analogous to execute.c, but requiring additional state describing a
 * persistent connection.
 */

// System includes
#include <errno.h>
#include <stdarg.h>
#include <stdbool.h>

// Project includes
#include <cuda_hidden.h>
#include <cuda/ops.h>
#include <debug.h>
#include <fatcubininfo.h>
#include <io/sock.h>
#include <method_id.h>
#include <packetheader.h>

/* NOTES
 *
 * Please read the notes in execute.c in addition to these.
 *
 * Some functions here require an additional argument; it is found at the
 * _second_ argument in the va_list (so third function argument), ignoring the
 * first:
 *
 * 		struct sockconn*
 *
 * Input and/or output pointer arguments in most functions have a known size via
 * use of well-defined data types, such as struct cudaDeviceProp. Others' sizes
 * can only be determined at runtime, such as the argument to
 * __cudaRegisterFatBinary() and of course all cudaMemcpy*() functions. However,
 * the latter specify the size in the argument list, allowing us to send the
 * packet first, then the data. This enables the remote end to deduce how much
 * data, if at all, it should expect. The function itself tells us also who
 * sends data: cudaGetDeviceProperties says the server sends a data portion, not
 * the client. Unfortunately, registerFatBinary does not have a size argument;
 * where that is is documented in the comments for that function within the
 * interposer.
 *
 * In other words, each function is implemented specifically for the call that
 * is made.
 *
 * The order of operations for each RPC below is tightly coupled with the
 * operation of remotesink.c:cudarpc_has_payload() and depends on how the
 * interposer packs the RPC to begin with.
 */

/*-------------------------------------- HIDDEN FUNCTIONS --------------------*/

// Any TODOs that are in execute.c also apply to this file.

/**
 * Pull out the network connection structure from the va_list given to the
 * function. It is found second in the va_list.
 *
 * @param conn		Pointer to struct sockconn*
 * @param argname	Name of last named parameter of function. Refer to
 * 					OPS_FN_PROTO.
 */
#define GET_CONN_VALIST(conn,argname)					\
	do {												\
		va_list extra;									\
		va_start(extra, argname);						\
		va_arg(extra, void*);	/* skip first arg */	\
		(conn) = va_arg(extra, struct sockconn*);		\
		BUG(!(conn));										\
		va_end(extra);									\
	} while(0)

/**
 * Examine 'err' (the return code for conn_*() calls). If indicative of failure,
 * set 'exit_errno' to -ENETDOWN and jump to label 'fail' in function. Just to
 * save some very repetitive typing.
 */
#define CONN_FAIL_ON_ERR(err)		\
	do { 							\
		if (err < 0) {				\
			exit_errno = -ENETDOWN;	\
			goto fail;				\
		}							\
	} while(0)

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(__CudaRegisterFatBinary)
{
	int err, exit_errno;
	void *cubin_marshaled = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t cubin_size = pkt->args[1].argll;

	printd(DBG_DEBUG, "cubin_size=%lu\n", cubin_size);

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	// the other side needs to maintain a fatcubins list for the driver local on
	// that machine
	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, cubin_marshaled, cubin_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt)); // return value captured in pkt
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(__CudaUnregisterFatBinary)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(__CudaRegisterFunction)
{
	int err, exit_errno;
	void *func_marshaled = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t func_size = pkt->args[1].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, func_marshaled, func_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt)); // return value captured in pkt
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(__CudaRegisterVar)
{
	int err, exit_errno;
	void *var_marshaled = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t var_size = pkt->args[1].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, var_marshaled, var_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt)); // return value captured in pkt
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaSetDevice)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaConfigureCall)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(CudaSetupArgument)
{
	int err, exit_errno;
	void *arg = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t arg_size = pkt->args[1].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, arg, arg_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaLaunch)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaThreadExit)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaThreadSynchronize)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaMalloc)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaFree)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(CudaMemcpyH2D)
{
	int err, exit_errno;
	void *data = (void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t data_size = pkt->args[2].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, data, data_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is pulled from the server.
static OPS_FN_PROTO(CudaMemcpyD2H)
{
	int err, exit_errno;
	void *data = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t data_size = pkt->args[2].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, data, data_size);
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has no payload.
static OPS_FN_PROTO(CudaMemcpyD2D)
{
	int err, exit_errno;

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is sent to the server.
static OPS_FN_PROTO(CudaMemcpyToSymbolH2D)
{
	int err, exit_errno;
	void *data = (void*)((uintptr_t)pkt + pkt->args[1].argull);
	size_t data_size = pkt->args[2].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_put(conn, data, data_size);
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}

// This function has a payload, which is pulled from the server.
static OPS_FN_PROTO(CudaMemcpyFromSymbolD2H)
{
	int err, exit_errno;
	void *data = (void*)((uintptr_t)pkt + pkt->args[0].argull);
	size_t data_size = pkt->args[2].arr_argi[0];

	struct sockconn *conn;
	GET_CONN_VALIST(conn,pkt);

	err = conn_put(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, pkt, sizeof(*pkt));
	CONN_FAIL_ON_ERR(err);
	err = conn_get(conn, data, data_size);
	CONN_FAIL_ON_ERR(err);

	return 0;
fail:
	return exit_errno;
}


const struct cuda_ops rpc_ops =
{
	// Functions which take only a cuda_packet*
	.configureCall = CudaConfigureCall,
	.free = CudaFree,
	.malloc = CudaMalloc,
	.memcpyD2D = CudaMemcpyD2D,
	.memcpyD2H = CudaMemcpyD2H,
	.memcpyH2D = CudaMemcpyH2D,
	.setDevice = CudaSetDevice,
	.setupArgument = CudaSetupArgument,
	.threadExit = CudaThreadExit,
	.threadSynchronize = CudaThreadSynchronize,
	.unregisterFatBinary = __CudaUnregisterFatBinary,

	// Functions which take a cuda_packet*, NULL then a sockconn*
	.launch = CudaLaunch,
	.memcpyFromSymbolD2H = CudaMemcpyFromSymbolD2H,
	.memcpyToSymbolH2D = CudaMemcpyToSymbolH2D,
	.registerFatBinary = __CudaRegisterFatBinary,
	.registerFunction = __CudaRegisterFunction,
	.registerVar = __CudaRegisterVar,
};

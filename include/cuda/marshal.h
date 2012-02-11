/**
 * @file marshal.h
 *
 * @date Mar 1, 2011
 * @author Magda Slawinska, magg@gatech.edu
 *
 * @date 2011-12-18
 * @author Alex Merritt, merritt.alex@gatech.edu
 * - Cleaned out obsolete functions
 */

#ifndef CITUILS_H_
#define CITUILS_H_

// CUDA includes
#include <__cudaFatFormat.h>

// Project includes
#include <cuda/fatcubininfo.h>
#include <cuda/packet.h>

#define OK 0
#define ERROR -1

/**
 * For storing the number of records  for particular structures
 * contained in the __cudaFatCubinBinaryRec
 */
typedef struct {
	int nptxs;
	int ncubs;
	int ndebs;
	int ndeps; // number of dependends
	int nelves;
	int nexps; // number of exported
	int nimps; // number of imported
} cache_num_entries_t;

/**
 * To use it when marshaling and unmarshaling in the sent packet
 * Should indicate the size of the following bytes
 */
typedef unsigned int size_pkt_field_t;

int mallocCheck(const void * const p, const char * const pFuncName,
		const char * pExtraMsg);

int getFatRecPktSize(const __cudaFatCudaBinary *pFatCubin,
		cache_num_entries_t * pEntriesCache);
//int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num);

int packFatBinary(char * pFatPack, __cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache);
int unpackFatBinary(__cudaFatCudaBinary *pFatC, char * pFatPack);

int getSize_regFuncArgs(void** fatCubinHandle, const char* hostFun,
        char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
        uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
int packRegFuncArgs(void *dst, void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
int unpackRegFuncArgs(reg_func_args_t * pRegFuncArgs, char * pPacket);

int getSize_regVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
		const char *deviceName, int ext, int vsize,int constant, int global);
int packRegVar(void *dst, void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int vsize, int
		constant, int global);
int unpackRegVar(reg_var_args_t * pRegVar, char *pPacket);

int freeRegFunc(reg_func_args_t *args);
int freeFatBinary(__cudaFatCudaBinary *fatCubin);
int freeRegVar(reg_var_args_t *args);

/**
 * cleans the structure, frees the allocated memory, sets values to zeros,
 * nulls, etc; intended to be used in __unregisterCudaFatBinary
 */
int cleanFatCubinInfo(struct cuda_fatcubin_info * pFatCInfo);

#endif /* CITUILS_H_ */

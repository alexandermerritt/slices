/**
 * @file libciutils.h
 * @brief The header file for the libciutils.c
 *
 * @date Mar 1, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#ifndef CITUILS_H_
#define CITUILS_H_

#include <__cudaFatFormat.h>
#include "packetheader.h"
#include "fatcubininfo.h"

/**
 * For storing the number of records  for particular structures
 * contained in the __cudaFatCubinBinaryRec
 */
typedef struct {
	int nptxs;
	int ncubs;
	int ndebs;
	int ndeps;		// number of dependends
	int nelves;
	int nexps;		// number of exported
	int nimps;		// number of imported
} cache_num_entries_t;

typedef struct {
	int * p;
	int size;
} array_int_t;

/**
 * To use it when marshaling and unmarshaling in the sent packet
 * Should indicate the size of the following bytes
 */
typedef unsigned int size_pkt_field_t;


int mallocCheck(const void * const p, const char * const pFuncName,
		const char * pExtraMsg);


int getFatRecPktSize(const __cudaFatCudaBinary *pFatCubin, cache_num_entries_t * pEntriesCache);
//int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num);

int packFatBinary(char * pFatPack, __cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache);
int unpackFatBinary(__cudaFatCudaBinary *pFatC, char * pFatPack);

char * packRegFuncArgs( void** fatCubinHandle, const char* hostFun,
        char* deviceFun, const char* deviceName, int thread_limit,
        uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize,
        int *pSize);
int unpackRegFuncArgs(reg_func_args_t * pRegFuncArgs, char * pPacket);


int freeRegFunc(reg_func_args_t *args);
int freeFatBinary(__cudaFatCudaBinary *fatCubin);

cuda_packet_t * callocCudaPacket(const char * pFunctionName, cudaError_t * pCudaError);

// print utilities
void l_printFatBinary(__cudaFatCudaBinary * pFatBin);
void l_printRegFunArgs(void** fatCubinHandle, const char* hostFun,
		char* deviceFun, const char* deviceName, int thread_limit, uint3* tid,
		uint3* bid, dim3* bDim, dim3* gDim, int* wSize);

/**
 * cleans the structure, frees the allocated memory, sets values to zeros,
 * nulls, etc; intended to be used in __unregisterCudaFatBinary
 */
int cleanFatCubinInfo(fatcubin_info_t * pFatCInfo);

/**
 * Translates method id to string
 * @param method_id The method id
 * @return a string corresponding to a given method id
 *         NULL if a method id has not been found
 */
char * methodIdToString(const int method_id);

#endif /* CITUILS_H_ */

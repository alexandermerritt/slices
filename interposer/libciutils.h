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

/**
 * For storing the number of records  for particular structures
 * contained in the __cudaFatCubinBinaryRec
 */
typedef struct {
	int nptxs;
	int ncubs;
	int ndebs;
	int nrecs;
	int nelves;
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
int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num);

__cudaFatCudaBinary * serializeFatBinary(__cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache,
		__cudaFatCudaBinary * const pDestFatC);

cuda_packet_t * callocCudaPacket(const char * pFunctionName, cudaError_t * pCudaError);

#endif /* CITUILS_H_ */

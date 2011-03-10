/**
 * @file libciutils.c
 * @brief Have some utils functions copied from cudart.c
 *
 * @date Mar 1, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#include <__cudaFatFormat.h>
#include <string.h>
#include "libciutils.h"
#include <stdio.h>
#include <stdlib.h>
#include "debug.h"


/**
 * checks if the memory has been appropriately allocated
 * @todo this function should go to a separate library
 * @param p (in) a pointer to a memory to be verified
 * @param pFuncName the name of the function when the problem occurred
 * @param pExtraMsg if you need to add an extra message
 *
 * @return 0 - pointer ok, -1 pointer not ok
 */
int mallocCheck(const void * const p, const char * const pFuncName,
		const char * pExtraMsg){
	if( NULL == p ){
		printd(DBG_ERROR, "%s: Problems with memory allocation. %s\n", pFuncName, pExtraMsg);
		return ERROR;
	}
	return OK;
}




/**
 * packs the fat cubin to the packe
 *
 * @param pFatCubin (in) the fat cubin binary to be copied; we assume that the
 *        packet
 * @param pPacket (out) the packet that will be packed in order to send it
 * 		  over
 * @return status of the operation: OK if everything went well, ERROR if something
 *        went wrong
 *
 */
//int packFatCubin(const __cudaFatCudaBinary * pFatCubin, cuda_packet_t * pPacket){
//	pPacket -> args[0].argp = nFatCubin;
//
//	nFatCubin->magic = fatCubin->magic;
//		nFatCubin->version = fatCubin->version;
//		nFatCubin->gpuInfoVersion = fatCubin->gpuInfoVersion ;
//}


/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatPtxEntry; includes the
 * size of the pointer to the structure
 *
 * @param pEntry (in) the entry we want to count
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatPtxEntry(const __cudaFatPtxEntry * pEntry, cache_num_entries_t * pEntriesCache){
	// apparently the __cudaFatPtxEntry might be an array
	// that's why we want to iterate through all this elements
	int i = 0;
	int size = 0;

	size += sizeof( __cudaFatPtxEntry *);  // space to store a pointer

	if (pEntry != NULL) {
		while (!(pEntry[i].gpuProfileName == NULL && pEntry[i].ptx == NULL)) {
			size += sizeof(__cudaFatPtxEntry ); // space to store elems the pointer
			size += (strlen(pEntry[i].gpuProfileName) + 1) * sizeof(char); // size of elements
			size += (strlen(pEntry[i].ptx) + 1) * sizeof(char);
			i++;
			pEntriesCache->nptxs++;
		}
		// Account for the null entries but no strlen required
		size += sizeof(__cudaFatPtxEntry ); // space to store elems at the addr
		pEntriesCache->nptxs++; // for the null entry (the end of 'array')
	}
	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatCubinEntry(const __cudaFatCubinEntry * pEntry, cache_num_entries_t * pEntriesCache){
	// apparently the __cudaFatCubinEntry might be an array
	// that's why we want to iterate through all this elements
	int i = 0;
	int size = 0;

	size += sizeof(__cudaFatCubinEntry *); // space to store addr

	if (pEntry != NULL) {
		while (!(pEntry[i].gpuProfileName == NULL && pEntry[i].cubin == NULL)) {
			size += sizeof(__cudaFatCubinEntry ); // space for the pointer to actual elements stored
			size += (strlen(pEntry[i].gpuProfileName) + 1) * sizeof(char); // size of elements
			size += (strlen(pEntry[i].cubin) + 1) * sizeof(char);
			pEntriesCache -> ncubs++;
			i++;
		}
		size += sizeof(__cudaFatCubinEntry ); // space to store elems at the addr
		pEntriesCache -> ncubs++;
	}

	return size;
}


/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @todo new stauff added in comparison to 1.1
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatDebugEntry(const __cudaFatDebugEntry * pEntry, cache_num_entries_t * pEntriesCache){

	// apparently the __cudaFatDebugEntry might be an array
	// that's why we want to iterate through all this elements
	int i = 0;
	int size = 0;

	size += sizeof( __cudaFatDebugEntry *);  // space to store the pointer

	if (pEntry != NULL) {
		while (!(pEntry[i].gpuProfileName == NULL && pEntry[i].debug == NULL)) {
			size += sizeof(__cudaFatDebugEntry ); // space to store elems at the addr
			size += (strlen(pEntry[i].gpuProfileName) + 1) * sizeof(char); // size of elements
			size += (strlen(pEntry[i].debug) + 1) * sizeof(char);
			// added stuff
			size += sizeof(__cudaFatDebugEntry *);
			size += sizeof(pEntry->size);
			pEntriesCache -> ndebs++;
			i++;
		}
		size += sizeof(__cudaFatDebugEntry ); // space to store elems at the addr
		pEntriesCache -> ndebs++;
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatSymbol imported; includes the
 * size of the pointer to the structure
 *
 * it should be done simply by size += sizeof( __cudaFatSymbol *), but
 * some empirical things came up and this is as it is done
 *
 * @param pEntry (in) the entry we want to count the size of
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatEntryExported(__cudaFatSymbol * pEntry){

	int size = 0;
	__cudaFatSymbol *p = pEntry;

	// it should be done simply by size += sizeof( __cudaFatSymbol *), but
	// some empirical things came up and this is as it is done
	//
	// there can be some issue with the ptr addr which can cause the code to crash
	// Therefore hacking (the while loop) instead of size += sizeof( __cudaFatSymbol *)
	while (p != NULL && p->name != NULL) {
		size += sizeof(__cudaFatSymbol *);  // space to store addr
		size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
		size += strlen(p->name) + 1;  // size of elements
		p++;
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatSymbol imported; includes the
 * size of the pointer to the structure
 *
 * it should be done simply by size += sizeof( __cudaFatSymbol *), but
 * some empirical things came up and this is as it is done
 *
 * @param pEntry (in) the entry we want to count the size of
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatEntryImported(__cudaFatSymbol * pEntry){

	int size = 0;
	__cudaFatSymbol *p = pEntry;

	// hacking again, instead of size += sizeof(__cudaFatSymbol *);
	// there can be some issue with the ptr addr which can cause the code to crash
	// Therefore hacking
	while (p != NULL && p->name != NULL) {
		size += sizeof(__cudaFatSymbol *);  // space to store addr
		size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
		size += strlen(p -> name) + 1;  // size of elements
		p++;
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @todo new stauff added in comparison to 1.1
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatBinaryEntry(__cudaFatCudaBinary * pEntry, cache_num_entries_t * pEntriesCache){

	// apparently the __cudaFatDebugEntry might be an array
	// that's why we want to iterate through all this elements
	int i = 0;
	int size = 0;

	// and again hacking instead of size += sizeof(__cudaFatCudaBinary *);  // space to store addr
	i = 0;
	if (pEntry != NULL) {
		while (pEntry[i].ident != NULL) {
			cache_num_entries_t nent = { 0, 0, 0, 0, 0 };
			size += sizeof(__cudaFatCudaBinary );
			size += getFatRecSize(&pEntry[i], &nent); // space to store elems at the addr
			pEntriesCache->nrecs++;
			i++;
		}
		size += sizeof(__cudaFatCudaBinary );
		pEntriesCache->nrecs++;
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @todo new stuff added in comparison to 1.1
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int _getSize__cudaFatElfEntry(const __cudaFatElfEntry * pEntry, cache_num_entries_t * pEntriesCache){

	// apparently the __cudaFatDebugEntry might be an array
	// that's why we want to iterate through all this elements
	int i = 0;
	int size = 0;

	size += sizeof( __cudaFatElfEntry *);  // space to store the pointer

	if (pEntry != NULL) {
		while (!(pEntry[i].gpuProfileName == NULL && pEntry[i].elf == NULL)) {
			size += sizeof(__cudaFatElfEntry ); // space to store elems at the addr
			size += (strlen(pEntry[i].gpuProfileName) + 1) * sizeof(char); // size of elements
			size += (strlen(pEntry[i].elf) + 1) * sizeof(char);
			// added stuff
			size += sizeof(__cudaFatElfEntry *);
			size += sizeof(pEntry[i].size);
			pEntriesCache -> nelves++;
			i++;
		}
		size += sizeof(__cudaFatElfEntry ); // space to store elems at the addr
		pEntriesCache -> nelves++;
	}

	return size;
}

/**
 * @brief gets the size of the fatcubin
 *
 * This is based on cuda 3.2 api /opt/cuda/include/__cudaFatFormat.h (the added
 * some fields in comparison to cuda 1.1, so you need it check it out when
 * you upgrade to the other version of cuda
 *
 * @todo update accordingly if you change cuda version you work with
 *
 * @param pFatCubin (in) this structure we want to compute the size
 * @param pEntriesCache (out) contains the sizes of entry structures in __cudaFatCudaBinary
 * @return the size of the fatcubin
 *
 */
int getFatRecSize(const __cudaFatCudaBinary *pFatCubin, cache_num_entries_t * pEntriesCache){

	int size = 0;

	size += sizeof(pFatCubin->magic);
	size += sizeof(pFatCubin->version);
	size += sizeof(pFatCubin->gpuInfoVersion);

	// The char * are supposed to be null terminated
	size += (strlen(pFatCubin->key) + 1) * sizeof(char);  // always 1 extra for the null char
	size += (strlen(pFatCubin->ident) + 1) * sizeof(char);
	size += (strlen(pFatCubin->usageMode) + 1) * sizeof(char);

	size += _getSize__cudaFatPtxEntry(pFatCubin->ptx, pEntriesCache);
	size += _getSize__cudaFatCubinEntry(pFatCubin->cubin, pEntriesCache);
	size += _getSize__cudaFatDebugEntry(pFatCubin->debug, pEntriesCache);

	size += sizeof(pFatCubin->debugInfo);
	size += sizeof(pFatCubin->flags);
	size += sizeof(pFatCubin->characteristic);

	size += _getSize__cudaFatEntryExported(pFatCubin->exported);
	size += _getSize__cudaFatEntryExported(pFatCubin->imported);

	//@todo check this
	size += _getSize__cudaFatBinaryEntry(pFatCubin->dependends, pEntriesCache);
	size += sizeof(pFatCubin->characteristic);

	size += _getSize__cudaFatElfEntry(pFatCubin -> elf, pEntriesCache);
	return size;
}

/**
 * allocates the cuda packet
 *
 * @param pFunctionName (in) the name of the function that called this allocation
 * @param pCudaError (out) sets cudaErrorMemoryAllocation if problems with allocating
 *        the packet appeared
 * @return pointer to the newly allocated cuda packet, NULL if memory allocation occurred
 */
cuda_packet_t * callocCudaPacket(const char * pFunctionName, cudaError_t * pCudaError){
	cuda_packet_t * packet = (cuda_packet_t *) calloc(1, sizeof(cuda_packet_t));
	if (packet == NULL) {
		printd(DBG_DEBUG, "%s, Problems with memory allocation for a cuda packet\n", pFunctionName);
		*pCudaError = cudaErrorMemoryAllocation;
	}
	return packet;
}

// serialization functions

// counts the new_marker; changes curr_marker and new_marker
#define GET_LOCAL_POINTER(curr_marker, size, new_marker, dtype) { \
	curr_marker = (char *)((unsigned long)curr_marker + size); \
	new_marker = dtype(curr_marker); \
}

#define COPY_STRINGS(nFat,fat,str,cur,l) { \
	strcpy(nFat->str, fat->str); \
	l = (strlen(fat->str) + 1) * sizeof(char); \
	cur[l - sizeof(char)] = 0; \
}

#define OFFSET_PTR_MEMB(ptr,memb,base,dtype) { \
	ptr->memb = dtype((unsigned long)ptr->memb - (unsigned long)base); \
}


/**
 * serializes __cudaFatPtxEntry
 *
 * @param pCurrent (inout) current position where we can serialize; should be
 *                 updated
 * @param pSrcFatC (in) the source from where can get the entry
 * @param pDestFatC (in) the destination where we serialize the entry
 * @param nEntries (in) the number of ptx records to be serialized
 *
 * @return OK
 * @todo this is how I want to have this done this big serialize
 */
/*int _serializeElfEntry(char ** pCurrent, __cudaFatCudaBinary * const pSrcFatC,
		__cudaFatCudaBinary * const pDestFatC,
		const int nEntries) {

	__cudaFatPtxEntry *pSrcEntry, *pDestEntry;
	int len, i;

	len = nEntries * sizeof(__cudaFatPtxEntry );
	pSrcEntry = pSrcFatC->ptx;
	pDestEntry = pDestFatC->ptx;
	if (pSrcEntry != NULL) {
		for (i = 0; i < nEntries; ++i) {
			if (pSrcEntry[i].gpuProfileName != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, pDestEntry[i].gpuProfileName, (char *));
				COPY_STRINGS((&pDestEntry[i]),(&pSrcEntry[i]),gpuProfileName,pCurrent,len);
				OFFSET_PTR_MEMB((&pDestEntry[i]),gpuProfileName,pDestFatC,(char *));
			} else
				pDestEntry[i].gpuProfileName = 0;
			if (pSrcEntry[i].ptx != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, pDestEntry[i].ptx, (char *));
				COPY_STRINGS((&pDestEntry[i]),(&pSrcEntry[i]),ptx,pCurrent,len);
				OFFSET_PTR_MEMB((&pDestEntry[i]),ptx,pDestFatC,(char *));
			} else
				pDestEntry[i].ptx = 0;
		}
		OFFSET_PTR_MEMB(pDestFatC,ptx,pDestFatC,(__cudaFatPtxEntry *));
	} else
		pDestFatC->ptx = NULL;

	return OK;




	GET_LOCAL_POINTER(*pCurrent, len, pDestFatC->debug, (__cudaFatDebugEntry *));
			len = pEntriesCache->ndebs * sizeof(__cudaFatDebugEntry);
			tempDeb = pSrcFatC->debug;
			nTempDeb = pDestFatC->debug;
			if (tempDeb != NULL) {
				for (i = 0; i < pEntriesCache->ndebs; ++i) {
					if (tempDeb[i].gpuProfileName != NULL) {
						GET_LOCAL_POINTER(pCurrent, len, nTempDeb[i].gpuProfileName, (char *));
						COPY_STRINGS((&nTempDeb[i]),(&tempDeb[i]),gpuProfileName,pCurrent,len);
						OFFSET_PTR_MEMB((&nTempDeb[i]),gpuProfileName,pDestFatC,(char *));
					}
					else
						nTempDeb[i].gpuProfileName = 0;
					if (tempDeb[i].debug != NULL) {
						GET_LOCAL_POINTER(pCurrent, len, nTempDeb[i].debug, (char *));
						COPY_STRINGS((&nTempDeb[i]),(&tempDeb[i]),debug,pCurrent,len);
						OFFSET_PTR_MEMB((&nTempDeb[i]),debug,pDestFatC,(char *));
					}
					else
						nTempDeb[i].debug = 0;
				}
				OFFSET_PTR_MEMB(pDestFatC,debug,pDestFatC,(__cudaFatDebugEntry *));
			}
			else
				pDestFatC->debug = NULL;


} */



/**
 * @brief Serializes the code and puts the whole fat binary  to a contiguous space
 * pointed by a pointer
 *
 * serializes the code and puts the whole fat binary structure under the address
 * the function previously has been named copyFatBinary but actually
 * there are two copyFatBinary functions (in local_api_wrapper.c you can
 * find a second one). Previously it was called cudart.c::copyFatBinary()
 *
 * First we reserve the copy the all fields of the structure __cudaFatCudaBinary,
 * then we copy strings, and all pointers etc. We end strings with 0.
 *
 * @param pSrcFatC (in) the pointer to the fat cubin
 * @param pEntriesCache (in) cached some numerical entries that have been counted by
 *        e.g. getFatSize
 * @param pDestFatC (in) where we have the assigned contiguous space we
 *        can copy to; we assume that the size has been appropriately allocated
 *        so there is space for areas pointed by pointers
 * @return the pointer to a serialized fat cubin
 *
 * @todo put those things into separate functions to make that function shorter
 */
__cudaFatCudaBinary * serializeFatBinary(__cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache,
		__cudaFatCudaBinary * const pDestFatC){

	// holds where we currently are in the serializing area
	char * pCurrent;

	// @todo likely we will not need them. but let's see
	__cudaFatPtxEntry *tempPtx, *nTempPtx;
	__cudaFatCubinEntry *tempCub, *nTempCub;
	__cudaFatDebugEntry *tempDeb, *nTempDeb;
	__cudaFatElfEntry *tempElf, *nTempElf;
	int len, i;

	// Now make a copy in a contiguous buffer
	// Doing it step by step because we need to allocate pointers
	// as we go and there isnt much that will change by using memcpy
	printd(DBG_DEBUG, "%s: nFatCubin addr = %p\n", __FUNCTION__, pDestFatC);
	pDestFatC->magic = pSrcFatC->magic;
	pDestFatC->version = pSrcFatC->version;
	pDestFatC->gpuInfoVersion = pSrcFatC->gpuInfoVersion ;

	// char*: key, ident, usageMode
	pCurrent = (char *)((unsigned long)pDestFatC);
	len = sizeof(__cudaFatCudaBinary);

	// \todo Some repeat work. Can cache these lengths
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->key, (char *));
	COPY_STRINGS(pDestFatC,pSrcFatC,key,pCurrent,len);
	// Adjust all the pointer variables to be offset from base
	OFFSET_PTR_MEMB(pDestFatC,key,pDestFatC,(char *));
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->ident, (char *));
	COPY_STRINGS(pDestFatC,pSrcFatC,ident,pCurrent,len);
	OFFSET_PTR_MEMB(pDestFatC,ident,pDestFatC,(char *));
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->usageMode, (char *));
	COPY_STRINGS(pDestFatC,pSrcFatC,usageMode,pCurrent,len);
	OFFSET_PTR_MEMB(pDestFatC,usageMode,pDestFatC,(char *));

	// Ptx block
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->ptx, (__cudaFatPtxEntry *));
	len = pEntriesCache->nptxs * sizeof(__cudaFatPtxEntry );
	tempPtx = pSrcFatC->ptx;
	nTempPtx = pDestFatC->ptx;
	if (tempPtx != NULL) {
		for (i = 0; i < pEntriesCache->nptxs; ++i) {
			if (tempPtx[i].gpuProfileName != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempPtx[i].gpuProfileName, (char *));
				COPY_STRINGS((&nTempPtx[i]),(&tempPtx[i]),gpuProfileName,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempPtx[i]),gpuProfileName,pDestFatC,(char *));
			} else
				nTempPtx[i].gpuProfileName = 0;
			if (tempPtx[i].ptx != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempPtx[i].ptx, (char *));
				COPY_STRINGS((&nTempPtx[i]),(&tempPtx[i]),ptx,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempPtx[i]),ptx,pDestFatC,(char *));
			} else
				nTempPtx[i].ptx = 0;
		}
		OFFSET_PTR_MEMB(pDestFatC,ptx,pDestFatC,(__cudaFatPtxEntry *));
	} else
		pDestFatC->ptx = NULL;


	// Cubin block
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->cubin, (__cudaFatCubinEntry *));
	len = pEntriesCache->ncubs * sizeof(__cudaFatCubinEntry);
	tempCub = pSrcFatC->cubin;
	nTempCub = pDestFatC->cubin;
	if (tempCub != NULL) {
		for (i = 0; i < pEntriesCache->ncubs; ++i) {
			if (tempCub[i].gpuProfileName != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempCub[i].gpuProfileName, (char *));
				COPY_STRINGS((&nTempCub[i]),(&tempCub[i]),gpuProfileName,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempCub[i]),gpuProfileName,pDestFatC,(char *));
			}
			else
				nTempCub[i].gpuProfileName = 0;
			if (tempCub[i].cubin != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempCub[i].cubin, (char *));
				COPY_STRINGS((&nTempCub[i]),(&tempCub[i]),cubin,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempCub[i]),cubin,pDestFatC,(char *));
			}
			else
				nTempCub[i].cubin = 0;
		}
		OFFSET_PTR_MEMB(pDestFatC,cubin,pDestFatC,(__cudaFatCubinEntry *));
	}
	else
		pDestFatC->cubin = NULL;

	// Debug block
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->debug, (__cudaFatDebugEntry *));
	len = pEntriesCache->ndebs * sizeof(__cudaFatDebugEntry);
	tempDeb = pSrcFatC->debug;
	nTempDeb = pDestFatC->debug;
	if (tempDeb != NULL) {
		for (i = 0; i < pEntriesCache->ndebs; ++i) {
			if (tempDeb[i].gpuProfileName != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempDeb[i].gpuProfileName, (char *));
				COPY_STRINGS((&nTempDeb[i]),(&tempDeb[i]),gpuProfileName,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempDeb[i]),gpuProfileName,pDestFatC,(char *));
			}
			else
				nTempDeb[i].gpuProfileName = 0;
			if (tempDeb[i].debug != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempDeb[i].debug, (char *));
				COPY_STRINGS((&nTempDeb[i]),(&tempDeb[i]),debug,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempDeb[i]),debug,pDestFatC,(char *));
			}
			else
				nTempDeb[i].debug = 0;
		}
		OFFSET_PTR_MEMB(pDestFatC,debug,pDestFatC,(__cudaFatDebugEntry *));
	}
	else
		pDestFatC->debug = NULL;

	pDestFatC->debugInfo = pSrcFatC->debugInfo;
	pDestFatC->flags = pSrcFatC->flags;
#ifndef CORRECTWAY
	pDestFatC->exported = pSrcFatC->exported;
	pDestFatC->imported = pSrcFatC->imported;
	pDestFatC->dependends = NULL;
#else
	GET_LOCAL_POINTER(pCurrent, (strlen(pSrcFatC->debug->debug) + 1), pDestFatC->exported, (__cudaFatSymbol *));
	GET_LOCAL_POINTER(pCurrent, sizeof(__cudaFatSymbol), pDestFatC->exported->name, (char *));
	strcpy(pDestFatC->exported->name, pSrcFatC->exported->name);

	GET_LOCAL_POINTER(pCurrent, (strlen(pSrcFatC->exported->name) + 1), pDestFatC->imported, (__cudaFatSymbol *));
	GET_LOCAL_POINTER(pCurrent, sizeof(__cudaFatSymbol), pDestFatC->imported->name, (char *));
	strcpy(pDestFatC->imported->name, pSrcFatC->imported->name);

	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->dependends, (__cudaFatCudaBinary *));
	len = pEntriesCache->nrecs * sizeof(__cudaFatCudaBinary);
	tempRec = pSrcFatC->dependends;
	nTempRec = pDestFatC->dependends;
	cache_num_entries_t nent = {0};
	if (tempRec != NULL) {
		// \todo This part definitely needs testing.
		for (i = 0; i < pEntriesCache->nrecs; ++i) {
			// \todo Right now, this is completely wrong. Every new
			// element will end up overwriting the previous one bec
			// copyFatBinary in this case does  not know where to
			// start new allocations from.
			GET_LOCAL_POINTER(pCurrent, len, pCurrent, (char *));
			int size = get_fat_rec_size(&tempRec[i], &nent);
			copyFatBinary(tempRec, &nent, &nTempRec[i]);
			len = size;
		}
		OFFSET_PTR_MEMB(pDestFatC,dependends,pDestFatC,(__cudaFatCudaBinary *));
	}
	else
		pDestFatC->dependends = NULL;  // common case
#endif

	pDestFatC->characteristic = pSrcFatC->characteristic;

	// elf block, added by Magda Slawinska
	GET_LOCAL_POINTER(pCurrent, len, pDestFatC->elf, (__cudaFatElfEntry *));
	len = pEntriesCache->nelves * sizeof(__cudaFatElfEntry );
	tempElf = pSrcFatC->elf;
	nTempElf = pDestFatC->elf;
	if (tempElf != NULL) {
		for (i = 0; i < pEntriesCache->nelves; ++i) {
			if (tempElf[i].gpuProfileName != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempElf[i].gpuProfileName, (char *));
				COPY_STRINGS((&nTempElf[i]),(&tempElf[i]),gpuProfileName,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempElf[i]),gpuProfileName,pDestFatC,(char *));
			} else
				nTempElf[i].gpuProfileName = 0;
			if (tempElf[i].elf != NULL) {
				GET_LOCAL_POINTER(pCurrent, len, nTempElf[i].elf, (char *));
				COPY_STRINGS((&nTempElf[i]),(&tempElf[i]),elf,pCurrent,len);
				OFFSET_PTR_MEMB((&nTempElf[i]),elf,pDestFatC,(char *));
			} else
				nTempElf[i].elf = 0;
		}
		OFFSET_PTR_MEMB(pDestFatC,elf,pDestFatC,(__cudaFatElfEntry *));
	} else
		pDestFatC->elf = NULL;

	return pDestFatC;
}

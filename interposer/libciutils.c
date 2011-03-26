/**
 * @file libciutils.c
 * @brief Have some utils functions copied from cudart.c; but some of them
 * refactored by Magic (MS)
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
 * returns the size of the packet for the string
 * |string_length|string|NULL|
 *
 * Specific situations:
 * |0|NULL|NULL|  indicates string==NULL
 * |0|''|NULL|    indicates string=""
 * |2|"12"|NULL|  indicates string="12"
 *
 * @param string
 * @return size of the packet for the string
 */
inline int l_getStringPktSize(const char const * string){
	int size = sizeof(size_pkt_field_t);

	size += sizeof(char); // for NULL termination

	if( string == NULL || strlen(string) == 0)
		size += sizeof(char);
	else
		size += strlen(string) * sizeof(char);

	return size;
}

/**
 * gets the size of packeting the __cudaFatCudaBinary -> __cudaFatPtxEntry;
 *
 * @param pEntry (in) the entry we want to count
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int l_getSize__cudaFatPtxEntry(const __cudaFatPtxEntry * pEntry, int * pCounter){
	// the __cudaFatPtxEntry is an array and is terminated with entries
	// that have NULL elements
	int size = 0;
	int i = 0;

	// to store the number of ptx entries
	size += sizeof(size_pkt_field_t);

	// size of the string, string, plus NULL terminator
	if (pEntry != NULL) {
		while (pEntry[i].gpuProfileName != NULL && pEntry[i].ptx != NULL) {
			size += l_getStringPktSize(pEntry[i].gpuProfileName);
			size += l_getStringPktSize(pEntry[i].ptx);
			i++;
		}
		*pCounter = i;
	}

	return size;
}

/**
 * gets the size of the packet for the __cudaFatCudaBinary -> __cudaFatCubinEntry
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int l_getSize__cudaFatCubinEntry(const __cudaFatCubinEntry * pEntry, int * pCounter){
	// the __cudaFatCubinEntry is an array and is terminated with entries
	// that have NULL elements
	int size = 0;
	int i = 0;

	// to store the number of entries
	size += sizeof(size_pkt_field_t);

	// size of the string, string, plus NULL terminator
	if (pEntry != NULL) {
		while (pEntry[i].gpuProfileName != NULL && pEntry[i].cubin != NULL) {
			size += l_getStringPktSize(pEntry[i].gpuProfileName) +
					l_getStringPktSize(pEntry[i].cubin);
			i++;
		}
	}
	*pCounter = i;

	return size;
}

/**
 * gets the size of the packet of __cudaFatCudaBinary -> __cudaFatDebugEntry
 *
 * numofdebentries|
 * sizeof string|gpuProfileName|Null|sizeofstring |debug|NULL|size(uint)|
 * sizeof string|gpuProfileName|Null|sizeofstring |debug|NULL|size(uint)|
 * ....
 * as many as numofdebentries
 *
 * @todo new stuff added in comparison to 1.1; need to be tested if this
 * is an array or a list
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size packet entry
 */
int l_getSize__cudaFatDebugEntry(__cudaFatDebugEntry * pEntry, int * pCounter){

	// apparently the __cudaFatDebugEntry might be an array
	// that's why we want to iterate through all this elements
	int size = 0;
	__cudaFatDebugEntry * p = pEntry;

	// to store the number of entries
	size += sizeof(size_pkt_field_t);

	// @todo Question: is this an array or a list (pEntry->next)? Let's assume
	// it is a list; we might be wrong;
	while( p != NULL ){
		size += l_getStringPktSize(p->gpuProfileName);
		size += l_getStringPktSize(p->debug);
		size += sizeof(p->size);

		(*pCounter) ++;

		p = p->next;
	}

	return size;
}

/**
 * gets the size of the packet of the __cudaFatCudaBinary -> __cudaFatSymbol
 * exported/imported;
 *
 * Again let's assume this is an array that ends with NULL symbol name
 *
 * |0|0|NULL|'1'| might indicate pEntry==NULL
 * |1|0|NULL|NULL| might indicate pEntry!=NULL && pEntry->name == NULL
 * |1|0|''|NULL| might indicate pEntry!=NULL && pEntry->name == ""
 * ....
 *
 * @param pEntry (in) the entry we want to count the size of
 * @param counter (out) a counter to count the entry symbols
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int l_getSize__cudaFatSymbolEntry(const __cudaFatSymbol * pEntry, int * pCounter){
	int size = 0;

	// to store the number of entries
	size += sizeof(size_pkt_field_t);     // counter

	if( pEntry == NULL || pEntry->name == NULL){
		size += l_getStringPktSize(NULL);
	} else {
		// let's assume this is an array of symbol names ended with NULL
		// and it is allowable that a name is NULL; but I may be wrong
		while( pEntry->name != NULL ){
			size += l_getStringPktSize(pEntry->name);
			(*pCounter)++;
			pEntry++;
		}
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @todo new stuff added in comparison to 1.1
 * @see comment to l_getSize__cudaFatEntryExported(__cudaFatSymbol*)
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pCounter (out) the counter for the number of dependants
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int l_getSize__cudaFatBinaryEntry(__cudaFatCudaBinary * pEntry, cache_num_entries_t * pEntriesCache){

	// do not understand this implementation, and I am following the
	// original implementation
	int size = sizeof(size_pkt_field_t);   // for the counter
	int i = 0;

	if( pEntry != NULL ){
		while (pEntry[i].ident != NULL) {
			cache_num_entries_t nent = { 0, 0, 0, 0, 0, 0, 0 };
			size += getFatRecPktSize(&pEntry[i], &nent); // space to store elems at the addr
			pEntriesCache->ndeps++;
			i++;
		}
	}

	return size;
}

/**
 * gets the size of the __cudaFatCudaBinary -> __cudaFatCubinEntry; includes the
 * size of the pointer to the structure
 *
 * @todo new stuff added in comparison to cuda 1.1
 * This is almost identical to @see l_getSize__cudaFatDebugEntry()
 *
 * @param pEntry (in) the entry we want to count the size of
 * @oaram pEntriesCache (out) the cache for storing entries about FatEntries structures
 * @return the size of the entry (including the size of the pointer to the structure)
 */
int l_getSize__cudaFatElfEntry(__cudaFatElfEntry * pEntry, int * pCounter){

	// apparently the __cudaFatElfEntry might be an array
	// that's why we want to iterate through all this elements
	int size = 0;
	__cudaFatElfEntry * p = pEntry;

	// to store the number of entries
	size += sizeof(size_pkt_field_t);

	// @todo Question: is this an array or a list (pEntry->next)? Let's assume
	// it is a list; we might be wrong;
	while (p != NULL) {
		size += l_getStringPktSize(p->gpuProfileName);
		size += l_getStringPktSize(p->elf);
		size += sizeof(p->size);

		(*pCounter)++;

		p = p->next;
	}

	return size;
}

/**
 * @brief gets the size of the packet that will contain a fatcubin
 *
 * This is based on cuda 3.2 api /opt/cuda/include/__cudaFatFormat.h (the added
 * some fields in comparison to cuda 1.1, so you need it check it out when
 * you upgrade to the other version of cuda
 *
 * originally this function was implemented as
 * int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num)
 * and serialization/deserialization was based on offsets, so I guess
 * in particular fields they stored the offsets in the packet that is
 * sent over somewhere. My approach is different:
 * |magic|version|gpuInfoVersion|flags|characteristics|
 * size_of_key|key ....|NULL|size_of_indent|indent ...|NULL|
 * size_of_usageMode|usageMode ....|NULL|
 * debugInfo|
 * num_of_ptx|ptx_entry1|ptx_entry2|...|ptx_entry_n|
 * num_of_cubin|cubin_entry1|cubin_entry2|cubin_entry_n|
 * ....
 *
 * @todo update accordingly if you change cuda version you work with
 *
 * @param pFatCubin (in) this structure we want to compute the size
 * @param pEntriesCache (out) contains the sizes of entry structures in __cudaFatCudaBinary
 * @return the size of the fatcubin
 *
 */
int getFatRecPktSize(const __cudaFatCudaBinary *pFatCubin, cache_num_entries_t * pEntriesCache){

	int size = 0;

	// so here is the story with the sizeof operator: sizeof(__cudaFatCudaBinary)
	// might not equal the sum of its members, counted individually
	// it might be greater because it may include internal and trailing
	// padding used to align the members of the structure or union on memory boundaries.
	// that's why I will not take sizeof(__cudaFatCudaBinary), besides I do not
	// want to send pointers, but the data there, so I need to prepare the structure
	// for containing the data not pointers to the data (since they are
	// meaningless on the remote machine)
	// 32bit vs. 64bit should not be a concern since I am sending a message
	// to the remote machine, and I need appropriate amount of space available
	// counting size algorithm to some extent determines the serialization
	// algorithm

	// first pack numbers (longs and ints), we will pack it as
	size += sizeof(pFatCubin->magic);
	size += sizeof(pFatCubin->version);
	size += sizeof(pFatCubin->gpuInfoVersion);
	size += sizeof(pFatCubin->flags);
	size += sizeof(pFatCubin->characteristic);

	// now deal with strings;strlen() doesn't include NULL terminator
	// we will store those characters as the size as returned by strlen
	// then the string terminated plus NULL included (it will make simpler
	// deserializing by strcpy
	size += l_getStringPktSize(pFatCubin->key);
	size += l_getStringPktSize(pFatCubin->ident);
	size += l_getStringPktSize(pFatCubin->usageMode);

	// this probably means the information where the debug information
	// can be found (eg. the name of the file with debug, or something)
	// @todo don't know what to do with this member, originally a size of the
	// pointer has been counted; doing the same; but this doesn't make
	// much sense to me anyway
	size += sizeof(pFatCubin->debugInfo);

	// ptx is an array
	size += l_getSize__cudaFatPtxEntry(pFatCubin->ptx, &pEntriesCache->nptxs);

	// cubin is an array, actually we will treat it the same as ptx entry
	size += l_getSize__cudaFatCubinEntry(pFatCubin->cubin, &pEntriesCache->ncubs);

	// it looks as it can be a list, but in the past version it was an array
	// a new fields were added in the comparison to the previous versions
	size += l_getSize__cudaFatDebugEntry(pFatCubin->debug, &pEntriesCache->ndebs);

	// it looks as it is a list, but it was added (it was absent in the past v.)
	// it is very similar to the debug.
	size += l_getSize__cudaFatElfEntry(pFatCubin->elf, &pEntriesCache->nelves);

	// symbol descriptor exported/imported, needed for __cudaFat binary linking
	size += l_getSize__cudaFatSymbolEntry(pFatCubin->exported, &pEntriesCache->nexps);
	size += l_getSize__cudaFatSymbolEntry(pFatCubin->imported, &pEntriesCache->nimps);

	size += l_getSize__cudaFatBinaryEntry(pFatCubin->dependends, pEntriesCache);

	return size;
}

int get_fat_rec_size(__cudaFatCudaBinary *fatCubin, cache_num_entries_t *num)
{
	int size = 0, i;
	__cudaFatPtxEntry *tempPtx;
	__cudaFatCubinEntry *tempCub;
	__cudaFatDebugEntry *tempDeb;
//	__cudaFatElfEntry * tempElf;
	__cudaFatSymbol *tempExp, *tempImp;
	__cudaFatCudaBinary *tempRec;

	// Adding all fields independently to make sure we get platform
	// dependent size of everything and remain agnostic to minor
	// data type changes for these fields
	size += sizeof(fatCubin->magic);
	size += sizeof(fatCubin->version);
	size += sizeof(fatCubin->gpuInfoVersion);
	// The char * are supposed to be null terminated
	size += (strlen(fatCubin->key) + 1) * sizeof(char);  // always 1 extra for the null char
	size += (strlen(fatCubin->ident) + 1) * sizeof(char);
	size += (strlen(fatCubin->usageMode) + 1) * sizeof(char);

	size += sizeof( __cudaFatPtxEntry *);  // space to store addr
	tempPtx = fatCubin->ptx;
	i = 0;
	if (tempPtx != NULL) {
		while (!(tempPtx[i].gpuProfileName == NULL && tempPtx[i].ptx == NULL)) {
			size += sizeof( __cudaFatPtxEntry);  // space to store elems at the addr
			size += (strlen(tempPtx[i].gpuProfileName) + 1) * sizeof(char);  // size of elements
			size += (strlen(tempPtx[i].ptx) + 1) * sizeof(char);
			i++;
			num->nptxs++;
		}
		// Account for the null entries but no strlen required
		size += sizeof( __cudaFatPtxEntry);  // space to store elems at the addr
		num->nptxs++;	// for the null entry
	}
	size += sizeof( __cudaFatCubinEntry *);  // space to store addr
	tempCub = fatCubin->cubin;
	i = 0;
	if (tempCub != NULL) {
		while (	!(tempCub[i].gpuProfileName == NULL && tempCub[i].cubin == NULL)) {
			size += sizeof( __cudaFatCubinEntry);  // space to store elems at the addr
			size += (strlen(tempCub[i].gpuProfileName) + 1) * sizeof(char);  // size of elements
			size += (strlen(tempCub[i].cubin) + 1) * sizeof(char);
			num->ncubs++;
			i++;
		}
		size += sizeof( __cudaFatCubinEntry);  // space to store elems at the addr
		num->ncubs++;
	}
	size += sizeof( __cudaFatDebugEntry *);  // space to store addr
	tempDeb = fatCubin->debug;
	i = 0;
	if (tempDeb != NULL) {
		while (!(tempDeb[i].gpuProfileName == NULL && tempDeb[i].debug == NULL)) {
			size += sizeof( __cudaFatDebugEntry);  // space to store elems at the addr
			size += (strlen(tempDeb[i].gpuProfileName) + 1) * sizeof(char);  // size of elements
			size += (strlen(tempDeb[i].debug) + 1) * sizeof(char);
			num->ndebs++;
			i++;
		}
		size += sizeof( __cudaFatDebugEntry *);  // space to store elems at the addr
		num->ndebs++;
	}

	size += sizeof(fatCubin->debugInfo);
	size += sizeof(fatCubin->flags);

	tempExp = fatCubin->exported;
#ifndef CORRECTWAY
	size += sizeof(__cudaFatSymbol *);  // space to store addr
#else  // there can be some issue with the ptr addr which can cause the code to crash
	// Therefore hacking
	while (tempExp != NULL && tempExp->name != NULL) {
		size += sizeof(__cudaFatSymbol *);  // space to store addr
		size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
		size += strlen(tempExp->name) + 1;  // size of elements
		tempExp++;
	}
#endif

	tempImp = fatCubin->imported;
#ifndef CORRECTWAY
	size += sizeof(__cudaFatSymbol *);  // space to store addr
#else  // there can be some issue with the ptr addr which can cause the code to crash
	// Therefore hacking
	while (tempImp != NULL && tempImp->name != NULL) {
		size += sizeof(__cudaFatSymbol *);  // space to store addr
		size += sizeof(__cudaFatSymbol);  // space to store elems at the addr
		size += strlen(tempImp->name) + 1;  // size of elements
		tempImp++;
	}
#endif

	tempRec = fatCubin->dependends;
#ifndef CORRECTWAY
	size += sizeof(__cudaFatCudaBinary *);  // space to store addr
#else
	i = 0;
	if (tempRec != NULL) {
		while (tempRec[i].ident != NULL) {
			cache_num_entries_t nent = {0};
			size += sizeof(__cudaFatCudaBinary);
			size += get_fat_rec_size(&tempRec[i], &nent);  // space to store elems at the addr
			num->ndeps++;
			i++;
		}
		size += sizeof(__cudaFatCudaBinary);
		num->ndeps++;
	}
#endif

	size += sizeof(fatCubin->characteristic);
/*		size += sizeof( __cudaFatElfEntry *);  // space to store addr
		tempElf = fatCubin->elf;
		i = 0;
		if (tempElf != NULL) {
			while (!(tempElf[i].gpuProfileName == NULL && tempElf[i].elf == NULL)) {
				size += sizeof( __cudaFatElfEntry);  // space to store elems at the addr
				size += (strlen(tempElf[i].gpuProfileName) + 1) * sizeof(char);  // size of elements
				size += (strlen(tempElf[i].elf) + 1) * sizeof(char);
				num->nelves++;
				i++;
			}
			size += sizeof( __cudaFatElfEntry*);  // space to store elems at the addr
			num->nelves++;
		} */
//	printd(DLEVEL1, "%s: ident=%s, size found=%d\n", fatCubin->ident, size);
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
/**
 * pack the fat cubin into a packet that can be transmitted
 * over the network
 */
int packFatBinary(char * pFatPack, __cudaFatCudaBinary * const pSrcFatC,
		cache_num_entries_t * const pEntriesCache){

	memcpy(pFatPack, (unsigned long*)&pSrcFatC->magic, sizeof(pSrcFatC->magic));
	pFatPack += sizeof(pSrcFatC->magic);
	memcpy(pFatPack, (unsigned long*)&pSrcFatC->version, sizeof(pSrcFatC->version));
	pFatPack += sizeof(pSrcFatC->version);
	memcpy(pFatPack, (unsigned long*)&pSrcFatC->gpuInfoVersion, sizeof(pSrcFatC->gpuInfoVersion));
	pFatPack += sizeof(pSrcFatC->gpuInfoVersion);
	memcpy(pFatPack, (unsigned int*)&pSrcFatC->flags, sizeof(pSrcFatC->flags));
	pFatPack += sizeof(pSrcFatC->flags);
	memcpy(pFatPack, (unsigned int*)&pSrcFatC->characteristic, sizeof(pSrcFatC->characteristic));
	pFatPack += sizeof(pSrcFatC->characteristic);

	return OK;
}

int unpackFatBinary(__cudaFatCudaBinary *pFatC, char * pFatPack){

	memcpy(&pFatC->magic, (unsigned long*)pFatPack, sizeof(pFatC->magic));
	pFatPack += sizeof(pFatC->magic);
	memcpy(&pFatC->version, (unsigned long*) pFatPack, sizeof(pFatC->version));
	pFatPack += sizeof(pFatC->version);
	memcpy(&pFatC->gpuInfoVersion, (unsigned long*) pFatPack, sizeof(pFatC->gpuInfoVersion));
	pFatPack += sizeof(pFatC->gpuInfoVersion);
	memcpy(&pFatC->flags, (unsigned int *) pFatPack, sizeof(pFatC->flags));
	pFatPack += sizeof(pFatC->flags);
	memcpy(&pFatC->characteristic, (unsigned int*) pFatPack, sizeof(pFatC->characteristic));
	pFatPack += sizeof(pFatC->characteristic);

	return OK;
}


__cudaFatCudaBinary * serializeFatBinary1(__cudaFatCudaBinary * const pSrcFatC,
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
	len = pEntriesCache->ndeps * sizeof(__cudaFatCudaBinary);
	tempRec = pSrcFatC->dependends;
	nTempRec = pDestFatC->dependends;
	cache_num_entries_t nent = {0};
	if (tempRec != NULL) {
		// \todo This part definitely needs testing.
		for (i = 0; i < pEntriesCache->ndeps; ++i) {
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

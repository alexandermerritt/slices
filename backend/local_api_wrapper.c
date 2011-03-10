/**
 * @file local_api_wrapper.c
 * @brief copied from remote_gpu/nvidia_backend/local_api_wrapper.c
 *
 * @date Mar 3, 2011
 * @author Adapted by Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#ifndef _GNU_SOURCE		  // needed for RTLD_NEXT
#	define _GNU_SOURCE  1 // needed for interposition
#endif

#include "packetheader.h"  // cuda_packet_t
#include <__cudaFatFormat.h>
#include <cuda.h>		// for CUDA_SUCCESS
#include <dlfcn.h>		// dlsym
#include "debug.h"
#include <string.h>

/******************Unlisted CUDA calls for state registration used later***************/

int __nvback_cudaRegisterFatBinary(cuda_packet_t *packet){
	__cudaFatCudaBinary * fatCubin = (__cudaFatCudaBinary *)packet->args[0].argui;

/*	static void** (*func)(void* fatC) = NULL;
	char *error;

	if (!func) {
		func = dlsym(RTLD_NEXT, "__cudaRegisterFatBinary");
		if ((error = dlerror()) != NULL) {
			printd(DBG_ERROR, "%s\n", error);
			exit(-1);
		}
	} */
	void ** fatCubinHandle = __cudaRegisterFatBinary(fatCubin);
	packet->ret_ex_val.handle = fatCubinHandle;

	return CUDA_SUCCESS;  // \todo more informative return values?
}

/**********************Helper functions for the unlisted calls************/
// talking in terms of offset
#define ALLOC_COPY_CHARS(nFat,fat,str,l) { \
	l = strlen(fat->str) + 1; \
	nFat->str = (char *)calloc(l, sizeof(char)); \
	strcpy(nFat->str, fat->str); \
}

#define OFFSET_PTR_MEMB(ptr,memb,base,dtype) { \
	ptr->memb = dtype((unsigned long)ptr->memb + (unsigned long)base); \
}


/**
 * modified by MS to add the elf field from fatCubin;
 * copied from gpu_remote/nvidia_backend/local_api_wrapper.c
 * this unpacks what has been packed
 *
 */
void *copyFatBinary(__cudaFatCudaBinary *fatCubin){
	int i, len;
	__cudaFatCudaBinary *tempRec, *nTempRec;
	__cudaFatPtxEntry *tempPtx, *nTempPtx;
	__cudaFatCubinEntry *tempCub, *nTempCub;
	__cudaFatDebugEntry *tempDeb, *nTempDeb;
	__cudaFatElfEntry * tempElf, *nTempElf;
	int nptxs = 0, ncubs = 0, ndebs = 0, nrecs = 0, nelves = 0;

	__cudaFatCudaBinary *nFatCubin = (__cudaFatCudaBinary *)calloc(1, sizeof(__cudaFatCudaBinary));

	nFatCubin->magic = fatCubin->magic;
	printd(DBG_DEBUG, "nFatCubin addr = %p, fatCubin = %p\n", nFatCubin, fatCubin);
	nFatCubin->version = fatCubin->version;
	nFatCubin->gpuInfoVersion = fatCubin->gpuInfoVersion ;
	printd(DBG_DEBUG, "old version=%ld, new=%ld\n", fatCubin->version, nFatCubin->version);
	printd(DBG_DEBUG, "old magic=%ld, new=%ld\n", fatCubin->magic, nFatCubin->magic);
	printd(DBG_DEBUG, "old version=%ld, new=%ld\n", fatCubin->gpuInfoVersion, nFatCubin->gpuInfoVersion);

	// Even though these pointers are really stored as offset, the offset
	// cannot be 0 by virtue of the structure being there,
	// unless it was explicitly assigned NULL. So this check is still valid
	if (fatCubin->key != NULL) {
		OFFSET_PTR_MEMB(fatCubin,key,fatCubin,(char *));
		ALLOC_COPY_CHARS(nFatCubin,fatCubin,key,len);
	}
	else
		nFatCubin->key = 0;
	if (fatCubin->ident != NULL) {
		OFFSET_PTR_MEMB(fatCubin,ident,fatCubin,(char *));
		ALLOC_COPY_CHARS(nFatCubin,fatCubin,ident,len);
	}
	else
		nFatCubin->ident = 0;
	printd(DBG_DEBUG, "ident = %s\n", fatCubin->ident);
	if (fatCubin->usageMode != NULL) {
		OFFSET_PTR_MEMB(fatCubin,usageMode,fatCubin,(char *));
		ALLOC_COPY_CHARS(nFatCubin,fatCubin,usageMode,len);
	}
	else
		nFatCubin->usageMode = 0;

	// Ptx block
	if (fatCubin->ptx != NULL) {
		OFFSET_PTR_MEMB(fatCubin,ptx,fatCubin,(__cudaFatPtxEntry *));
		tempPtx = fatCubin->ptx;
		i = 0;
		while (!(tempPtx[i].gpuProfileName == NULL && tempPtx[i].ptx == NULL)) {
			i++;
			nptxs++;
		}
		nptxs++;	// for the null entry
		len = nptxs * sizeof(__cudaFatPtxEntry);

		nFatCubin->ptx = (__cudaFatPtxEntry *)malloc(len);
		nTempPtx = nFatCubin->ptx;
		for (i = 0; i < nptxs; ++i) {
			if (tempPtx[i].gpuProfileName != NULL) {
				OFFSET_PTR_MEMB((&tempPtx[i]),gpuProfileName,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempPtx[i]),(&tempPtx[i]),gpuProfileName,len);
			}
			else
				nTempPtx[i].gpuProfileName = 0;
			if (tempPtx[i].ptx != NULL) {
				OFFSET_PTR_MEMB((&tempPtx[i]),ptx,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempPtx[i]),(&tempPtx[i]),ptx,len);
			}
			else
				nTempPtx[i].ptx = 0;
		}
	}
	else
		nFatCubin->ptx = NULL;

	// Cubin block
	if (fatCubin->cubin != NULL) {
		OFFSET_PTR_MEMB(fatCubin,cubin,fatCubin,(__cudaFatCubinEntry *));
		tempCub = fatCubin->cubin;
		i = 0;
		while (!(tempCub[i].gpuProfileName == NULL && tempCub[i].cubin == NULL)) {
			i++;
			ncubs++;
		}
		ncubs++;	// for the null entry
		len = ncubs * sizeof(__cudaFatCubinEntry);

		nFatCubin->cubin = (__cudaFatCubinEntry *)malloc(len);
		nTempCub = nFatCubin->cubin;
		for (i = 0; i < ncubs; ++i) {
			if (tempCub[i].gpuProfileName != NULL) {
				OFFSET_PTR_MEMB((&tempCub[i]),gpuProfileName,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempCub[i]),(&tempCub[i]),gpuProfileName,len);
			}
			else
				nTempCub[i].gpuProfileName = 0;
			if (tempCub[i].cubin != NULL) {
				OFFSET_PTR_MEMB((&tempCub[i]),cubin,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempCub[i]),(&tempCub[i]),cubin,len);
			}
			else
				nTempCub[i].cubin = 0;
		}
	}
	else
		nFatCubin->cubin = NULL;

	// Debug block
	if (fatCubin->debug != NULL) {
		OFFSET_PTR_MEMB(fatCubin,debug,fatCubin,(__cudaFatDebugEntry *));
		tempDeb = fatCubin->debug;
		i = 0;
		while (!(tempDeb[i].gpuProfileName == NULL && tempDeb[i].debug == NULL)) {
			i++;
			ndebs++;
		}
		ndebs++;	// for the null entry
		len = ndebs * sizeof(__cudaFatDebugEntry);

		nFatCubin->debug = (__cudaFatDebugEntry *)malloc(len);
		nTempDeb = nFatCubin->debug;
		for (i = 0; i < ndebs; ++i) {
			if (tempDeb[i].gpuProfileName != NULL) {
				OFFSET_PTR_MEMB((&tempDeb[i]),gpuProfileName,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempDeb[i]),(&tempDeb[i]),gpuProfileName,len);
			}
			else
				nTempDeb[i].gpuProfileName = 0;
			if (tempDeb[i].debug != NULL) {
				OFFSET_PTR_MEMB((&tempDeb[i]),debug,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempDeb[i]),(&tempDeb[i]),debug,len);
			}
			else
				nTempDeb[i].debug = 0;
		}
	}
	else
		nFatCubin->debug = NULL;

	nFatCubin->debugInfo = fatCubin->debugInfo;
	nFatCubin->flags = fatCubin->flags;
#ifndef CORRECTWAY
	nFatCubin->exported = fatCubin->exported;
	nFatCubin->imported = fatCubin->imported;
#else
	printf("%s: NOT IMPLEMENTED YET\n", __FUNCTION__);
#endif

	if (fatCubin->dependends != NULL) {
		OFFSET_PTR_MEMB(fatCubin,dependends,fatCubin,(__cudaFatCudaBinary *));
		i = 0;
		tempRec = fatCubin->dependends;
		while (tempRec[i].ident != NULL) {
			nrecs++;
			i++;
		}
		nrecs++;
		len = nrecs * sizeof(__cudaFatCudaBinary);
		nFatCubin->dependends = (__cudaFatCudaBinary *)malloc(len);
		nTempRec = nFatCubin->dependends;
		// \todo This part definitely needs testing.
		for (i = 0; i < nrecs; ++i) {
			// \todo Right now, this is completely wrong. Every new
			// element will end up overwriting the previous one bec
			// copyFatBinary in this case does  not know where to
			// start new allocations from.
//			copyFatBinary(tempRec, &nTempRec[i]);
			printf("%s: This is totally wrong. Correct this before using!!!!!!!!!\n",
					__FUNCTION__);
		}
	}
	else
		nFatCubin->dependends = NULL;  // common case

	nFatCubin->characteristic = fatCubin->characteristic;

	// added by Magda Slawinska
	// elf entry
	if (fatCubin->elf != NULL) {
		OFFSET_PTR_MEMB(fatCubin,elf,fatCubin,(__cudaFatElfEntry *));
		tempElf = fatCubin->elf;
		i = 0;
		while (!(tempElf[i].gpuProfileName == NULL && tempElf[i].elf == NULL)) {
			i++;
			nelves++;
		}
		nelves++;	// for the null entry
		len = nelves * sizeof(__cudaFatElfEntry);

		nFatCubin->elf = (__cudaFatElfEntry *)malloc(len);
		nTempElf = nFatCubin->elf;
		for (i = 0; i < nelves; ++i) {
			if (tempElf[i].gpuProfileName != NULL) {
				OFFSET_PTR_MEMB((&tempElf[i]),gpuProfileName,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempElf[i]),(&tempElf[i]),gpuProfileName,len);
			}
			else
				nTempElf[i].gpuProfileName = 0;
			if (tempElf[i].elf != NULL) {
				OFFSET_PTR_MEMB((&tempElf[i]),elf,fatCubin,(char *));
				ALLOC_COPY_CHARS((&nTempElf[i]),(&tempElf[i]),elf,len);
			}
			else
				nTempElf[i].elf = 0;
		}
	}
	else
		nFatCubin->elf = NULL;


	return nFatCubin;
} // end copyFatBinary

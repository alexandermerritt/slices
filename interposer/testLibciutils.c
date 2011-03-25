/**
 * @file testLibciutils.c
 * @brief
 *
 * @date Mar 24, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CUnit/CUnit.h>
#include <CUnit/Basic.h>

#include "packetheader.h"
#include "debug.h"	// ERROR, OK
#include <pthread.h>

#include <__cudaFatFormat.h> // for __cudaFatPtxEntry, and others
#include "libciutils.h"	     // for cache_num_entries_t

extern inline int l_getStringPktSize(const char const * string);
extern int l_getSize__cudaFatPtxEntry(const __cudaFatPtxEntry * pEntry, cache_num_entries_t * pEntriesCache);
extern int l_getSize__cudaFatCubinEntry(const __cudaFatCubinEntry * pEntry, cache_num_entries_t * pEntriesCache);
extern int l_getSize__cudaFatDebugEntry(const __cudaFatDebugEntry * pEntry, cache_num_entries_t * pEntriesCache);

/* The suite initialization function.
 * Opens the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int init_suite1(void)
{
  /* if (NULL == (temp_file = fopen("temp.txt", "w+"))) {
      return -1;
   }
   else {
      return 0;
   }*/
   return 0;
}

/* The suite cleanup function.
 * Closes the temporary file used by the tests.
 * Returns zero on success, non-zero otherwise.
 */
int clean_suite1(void)
{
   /*if (0 != fclose(temp_file)) {
      return -1;
   }
   else {
      temp_file = NULL;
      return 0;
   }*/
  return 0;
}
/**
 * tests if the packet for a string is correctly counted
 */
void test_l_getStringPktSize(void){
	char * s;
	int size;

	// 0. null string NULL + NULL
	s = NULL;
	size = sizeof(size_pkt_field_t) + (1+1) * sizeof(char);
	CU_ASSERT( l_getStringPktSize(s) == size );

	// 1. zero length string
	s = "";
	// "" + NULL
	size = sizeof(size_pkt_field_t) + (1+1) *sizeof(char);
	CU_ASSERT( l_getStringPktSize(s) == size );

	// 2. regular string
	s = "1323";
	size = sizeof(size_pkt_field_t) + (4 + 1) * sizeof(char);
	CU_ASSERT( l_getStringPktSize(s) == size );
}

/**
 * tests if the packet for the ptxEntry is correctly counted
 */
void test_l_getSize__cudaFatPtxEntry(void){

	__cudaFatPtxEntry e1[] = { {NULL, NULL} };
	cache_num_entries_t cache = {0,0,0,0,0};
	int size = 0;

	// 0a. null entry
	size = sizeof(size_pkt_field_t);
	CU_ASSERT( l_getSize__cudaFatPtxEntry(NULL, &cache) == size);
	CU_ASSERT( 0 ==  cache.nptxs);

	// 0b. null values
	size = sizeof(size_pkt_field_t);
	cache.nptxs = 0;
	CU_ASSERT( l_getSize__cudaFatPtxEntry(e1, &cache) == size);
	CU_ASSERT( 0 == cache.nptxs);

	// 1. count the size of the empty structure
	__cudaFatPtxEntry e2[] = { {"", ""}, {NULL, NULL}};
	cache.nptxs = 0;
	size = sizeof(size_pkt_field_t)
			+ sizeof(size_pkt_field_t) + 1 + 1
			+ sizeof(size_pkt_field_t) + 1 + 1;
	CU_ASSERT( l_getSize__cudaFatPtxEntry(e2, &cache) ==  size);
	CU_ASSERT( 1 == cache.nptxs);

	// 2. put some not empty things
	__cudaFatPtxEntry e3[] = { {"1", "12"}, {"3", "7"}, {NULL, NULL} };
	cache.nptxs = 0;
	size = sizeof(size_pkt_field_t) + 4*sizeof(size_pkt_field_t) + (4+ 5) * sizeof(char);
	CU_ASSERT( l_getSize__cudaFatPtxEntry(e3, &cache) ==  size);
	CU_ASSERT( 2 == cache.nptxs);
}

/**
 * tests if the packet for the cubinEntry is correctly counted
 * Should be almost identical to @see test_l_getSize__cudaFatPtxEntry()
 */
void test_l_getSize__cudaFatCubinEntry(void){

	__cudaFatCubinEntry e1[] = { {NULL, NULL} };
	cache_num_entries_t cache = {0,0,0,0,0};
	int size = 0;

	// 0a. null entry
	size = sizeof(size_pkt_field_t);
	CU_ASSERT( l_getSize__cudaFatCubinEntry(NULL, &cache) == size);
	CU_ASSERT( 0 ==  cache.ncubs);

	// 0b. null values
	size = sizeof(size_pkt_field_t);
	cache.ncubs = 0;
	CU_ASSERT( l_getSize__cudaFatCubinEntry(e1, &cache) == size);
	CU_ASSERT( 0 == cache.ncubs);

	// 1. count the size of the empty structure
	__cudaFatCubinEntry e2[] = { {"", ""}, {NULL, NULL}};
	cache.ncubs = 0;
	size = sizeof(size_pkt_field_t)
			+ sizeof(size_pkt_field_t) + 1 + 1
			+ sizeof(size_pkt_field_t) + 1 + 1;
	CU_ASSERT( l_getSize__cudaFatCubinEntry(e2, &cache) ==  size);
	CU_ASSERT( 1 == cache.ncubs);


	// 3. put some not empty things |ncubs|(len|str|NULL|)*4
	__cudaFatCubinEntry e3[] = { {"1", "12"}, {"3", "7"}, {NULL, NULL} };
	// 5 chars
	cache.ncubs = 0;
	size = sizeof(size_pkt_field_t)
			+ 4*(sizeof(size_pkt_field_t) + sizeof(char))
			+ 5 * sizeof(char);
	CU_ASSERT( l_getSize__cudaFatCubinEntry(e3, &cache) ==  size);
	CU_ASSERT( cache.ncubs == 2);
}

/**
 * tests if the packet for the debugEntry is correctly counted
 *
 */
void test_l_getSize__cudaFatDebugEntry(){
	__cudaFatDebugEntry e[] = { {NULL, NULL, NULL, 0} };
	cache_num_entries_t cache = {0,0,0,0,0};
	int size;

	// 0. null entry
	size = sizeof(size_pkt_field_t);
	CU_ASSERT( l_getSize__cudaFatDebugEntry(NULL, &cache) == size);
	CU_ASSERT( 0 == cache.ndebs);

	// 1a. next null, the strings null
	cache.ndebs = 0;
	size = sizeof(size_pkt_field_t)
		+ sizeof(size_pkt_field_t) + 1 + 1
		+ sizeof(size_pkt_field_t) + 1 + 1 + sizeof(e[0].size);
	CU_ASSERT( l_getSize__cudaFatDebugEntry(e, &cache) == size);
	CU_ASSERT( 1 == cache.ndebs);

	// 1b. next is null, the strings are empty
	e[0].debug = "";
	e[0].gpuProfileName = "";
	e[0].next = NULL;
	cache.ndebs = 0;
	size = sizeof(size_pkt_field_t)
			+ sizeof(size_pkt_field_t) + 1 + 1
			+ sizeof(size_pkt_field_t) + 1 + 1
			+ sizeof(e[0].size);
	CU_ASSERT( l_getSize__cudaFatDebugEntry(e, &cache) == size);
	CU_ASSERT( 1 == cache.ndebs);

	// 2. next is null, the rest is not
	e[0].debug = "file1";
	e[0].gpuProfileName = "name";
	e[0].next = NULL;
	cache.ndebs = 0;
	size = sizeof(size_pkt_field_t) +
		   sizeof(size_pkt_field_t) + 5 + 1 +
		   sizeof(size_pkt_field_t) + 4 + 1 +
		   sizeof(e[0].size);
	CU_ASSERT( l_getSize__cudaFatDebugEntry(e, &cache) == size);
	CU_ASSERT( 1 == cache.ndebs);

	// 3. now normal situation
	__cudaFatDebugEntry e1[] = { {"profile", "debug", &e[0], 0}};

	size =  sizeof(size_pkt_field_t) +
			sizeof(size_pkt_field_t) + 7 + 1 +
			sizeof(size_pkt_field_t) + 5 + 1 +
			sizeof(e[1].size) +
			sizeof(size_pkt_field_t) + 5 + 1 +
			sizeof(size_pkt_field_t) + 4 + 1 +
			sizeof(e[0].size);
	cache.ndebs = 0;
	CU_ASSERT( l_getSize__cudaFatDebugEntry(e1, &cache) == size);
	CU_ASSERT( 2 == cache.ndebs);
}

/* The main() function for setting up and running the tests.
 * Returns a CUE_SUCCESS on successful running, another
 * CUnit error code on failure.
 */
int main()
{
   CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Suite_2", init_suite1, clean_suite1);
   if (NULL == pSuite) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if ( (NULL == CU_add_test(pSuite, "test of test_l_getStringPktSize", test_l_getStringPktSize)) ||
		(NULL == CU_add_test(pSuite, "test of test_l_getSize__cudaFatPtxEntry", test_l_getSize__cudaFatPtxEntry)) ||
        (NULL == CU_add_test(pSuite, "test of test_l_getSize__cudaFatCubinEntry", test_l_getSize__cudaFatCubinEntry)) ||
        (NULL == CU_add_test(pSuite, "test of test_l_getSize__cudaFatDebugEntry", test_l_getSize__cudaFatDebugEntry)) ){
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
}

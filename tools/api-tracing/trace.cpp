/**
 * @file trace.cpp
 *
 * @date March 20, 2012
 * @author Alex Merritt, merritt.alex@gatech.edu
 */

// C System includes
#include <string.h>

// C++ includes
#include <iostream>
#include <set>

// Directory-immediate includes
#include "trace.h"

/* Globals */

counter_t counter;
trace_t trace;

std::set<std::string> memcpy_names;

/* Functions */

void trace_init(void)
{
	counter.clear();
	memcpy_names.clear();
	memcpy_names.insert("cudaMemcpy");
	memcpy_names.insert("cudaMemcpyAsync");
	memcpy_names.insert("cudaMemcpyToSymbol");
	memcpy_names.insert("cudaMemcpyFromSymbol");
	memcpy_names.insert("cudaMemcpyToSymbolAsync");
	memcpy_names.insert("cudaMemcpyToArray");
}

void trace_report(void)
{
	std::cout << std::endl << ">>> TRACE REPORT" << std::endl;

	std::cout << "Count\tFunction" << std::endl;
	for(auto iter = counter.begin(); iter != counter.end(); iter++)
		std::cout << iter->second.count << "\t" << iter->first << std::endl;

	std::cout << "Call#\tFunction" << std::endl;
	unsigned int item = 1;
	struct call *c;
	for(auto iter = trace.begin(); iter != trace.end(); iter++) {
		std::cout << item++ << "\t" << std::get<0>(*iter);
		c = &std::get<1>(*iter);
		if(func_is_memcpy(std::get<0>(*iter)))
			std::cout << "\tbytes: " << c->bytes;
		std::cout << std::endl;
	}
}

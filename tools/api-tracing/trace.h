/**
 * @file trace.h
 *
 * @date March 20, 2012
 * @author Alex Merritt, merritt.alex@gatech.edu
 */

// C++ includes
#include <iostream>
#include <list>
#include <map>
#include <tuple>
#include <set>

/**
 * XXX HACK XXX
 * TRACE_PREOPS accesses a variable called 'count'. If that does not exist in
 * the function (and it wouldn't if the function is not a memcpy variant), the
 * compiler will complain. So, for functions which do not have it, I provide a
 * global to link against which is set to zero.  Functions which DO have a local
 * variable with the same name will reference the parameter instead of this
 * global (due to rules of scope visibility). Functions which are NOT a memcpy
 * but which DO have a count argument will generate a compiler warning if they
 * are of different types (e.g. cudaGetDeviceCount) but the if-conditional
 * protects against accessing that data.
 */
static size_t count = 0UL;

#define func_is_memcpy(f) \
	(memcpy_names.find(f) != memcpy_names.end())

#define TRACE_PREOPS \
{ \
	counter[__func__].count++; \
	struct call _c = { 0, 0UL }; \
	if (func_is_memcpy(__func__)) \
		_c.bytes = count; \
	trace.push_back(std::make_tuple(__func__, _c)); \
	/* TODO Latency */ \
}

#define TRACE_POSTOPS			\
{								\
	/* TODO Latency */ \
}

/* Data structures for aggregating information associated with each function. */
struct stats { unsigned long count; };
typedef std::map<std::string, struct stats> counter_t;

/* Data structures for composing the applications API trace. */
struct call
{
	size_t bytes; /* Associated data movement (e.g. cudaMemcpy) */
	unsigned long time; /* Latency of call */
};
typedef std::tuple<std::string, struct call> call_t;
typedef std::list<call_t> trace_t;

/* Objects pulled out from trace.cpp. */
extern counter_t counter;
extern trace_t trace;
extern std::set<std::string> memcpy_names;

void trace_init(void);
void trace_report(void);

/**
 * @file debug.h
 * @brief Contains debug utils
 * Copied from: remote_gpu/common/gvim_debug.h
 *
 * @date Feb 23, 2011
 * @author Modified and extended by Magda Slawinska, magg@gatech.edu
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

//////////////////////////////////////
//
// To enable/disable debug printing globally, pass 'dbg=1' to the compiler. That
// will define the macro DEBUG. Not specifying this actually preprocesses OUT
// all printd statements in code, reducing its size and the runtime overhead
// associated with it.

//////////////////////////////////////
//
// Debug printing verbosity, when enabled
//
#define DBG_LEVEL   DBG_DEBUG

#define DBG_ERROR   0	// defunct system
#define DBG_WARNING 1	// potentially defunct, perhaps unstable
#define DBG_INFO    2	// function flow, useful messages
#define DBG_DEBUG   3	// detailed: variable state, func args, etc

/**
 * Halt immediately if expression 'expr' evaluates to true and print a message.
 * Must be used where code is NOT expected to fail. Bad examples of its use
 * include checking return codes to networking functions and the like (we cannot
 * control failure of such functions). A good example is checking for specific
 * return values, where you wrote both the function itself, and the code which
 * invokes it.
 */
#define BUG(expr)						\
	do {								\
		if (expr) {						\
			printd(DBG_ERROR, "BUG\n");	\
			assert(0);					\
		}								\
	} while(0)							\

#define printd(level, fmt, args...)                                     \
    do {                                                                \
        if((level) <= DBG_LEVEL) {                                      \
            printf("<%d> (%d:%d) %s::%s[%d]: ",							\
					(level), getpid(),(pid_t)syscall(SYS_gettid),		\
					__FILE__, __func__, __LINE__);   					\
            printf(fmt, ##args);                                        \
            fflush(stdout);                                             \
        }                                                               \
    } while(0)

/**
 * If DEBUG is not defined, the preprocessor will remove from the code all
 * printd statements. To use printd, one must include this file. In doing so,
 * one also includes this statement which will leave in or actively remove
 * printd statements based on whether DEBUG is defined or not.
 */
#ifndef DEBUG
 #undef printd
 #define printd(l,f,a...)
#endif

#endif /* DEBUG_H_ */

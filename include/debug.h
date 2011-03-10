/**
 * @file debug.h
 * @brief Contains debug utils
 * Copied from: remote_gpu/common/gvim_debug.h
 *
 * @date Feb 23, 2011
 * @author Modified by Magda Slawinska, magg@gatech.edu
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include <stdio.h>

#define DEBUG

//////////////////////////////////////
//
// Debug printing verbosity
//
#define DBG_LEVEL   DBG_DEBUG

// New debug messaging state. There is no sense of a "level" for debugging. Each of these define the
// purpose of the messages and is enabled/disabled per file
#define DBG_ERROR   0           // system cannot continue
#define DBG_WARNING 1           // system is (may be?) consistent, stuff still works (e.g. accept no more domains)
#define DBG_INFO    2           // messages about state or configuration; high-level flow
#define DBG_DEBUG   3           // func args, variable values, etc; full flow, may slow system down

#define printd(level, fmt, args...)                                     \
    do {                                                                \
        if((level) <= DBG_LEVEL) {                                      \
            printf("<%d> %s[%d]: ", (level), __FUNCTION__, __LINE__);   \
            printf(fmt, ##args);                                        \
            fflush(stdout);                                             \
        }                                                               \
    } while(0)

//! return status when everything went ok
#define OK 0

//! return status when something went wrong
#define ERROR -1
#endif /* DEBUG_H_ */

/**
 * @file timing.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 2, 2012
 * @brief Definitions and functions for timing the runtime (not the CUDA API).
 */

#include <pthread.h>

#include <util/timer.h>
#include <util/compiler.h>

#include "timing.h"

#ifdef TIMING

/*-------------------------------------- INTERNAL STATE ----------------------*/

// Timer resolution set by TIMER_ACCURACY in util/timer.h

#if defined(TIMING)
static struct timer attach_timer; //! Timer used to measure joining the runtime
static uint64_t attach = 0UL; //! Time spent joining the runtime

static struct timer detach_timer; //! Timer used to measure departing the runtime
static uint64_t detach = 0UL; //! Time spent departing the runtime

static struct rpc_latencies lat; //! Aggregate latencies over all RPCs

// XXX NONE OF THESE ARE THREAD SAFE
#define TRACE_MAX_CALLS		(8UL << 20)
static unsigned long num_calls_made = 0UL;
/** Tallying all calls made into the API. */
static unsigned long api_counts[CUDA_METHOD_LIMIT];
/** Aggregate latencies for each function. */
static unsigned long api_lat[CUDA_METHOD_LIMIT];
static struct call trace[TRACE_MAX_CALLS];
#endif

/*-------------------------------------- FUNCTIONS ---------------------------*/

/** Initialize global timers above. */
void
timers_init(void)
{
	timer_init(CLOCK_REALTIME, &attach_timer);
	timer_init(CLOCK_REALTIME, &detach_timer);
}

void
timers_start_attach(void)
{
	timer_start(&attach_timer);
}

void
timers_stop_attach(void)
{
	attach = timer_end(&attach_timer, TIMER_ACCURACY);
}
void
timers_start_detach(void)
{
	timer_start(&detach_timer);
}

void
timers_stop_detach(void)
{
	detach = timer_end(&detach_timer, TIMER_ACCURACY);
}

inline void
trace_timestamp(void)
{
    clock_gettime(CLOCK_REALTIME, &trace[num_calls_made].ts);
}

/**
 * Called at the end of every CUDA call made in the interposer.
 *
 * XXX This function is in NO WAY thread-safe.
 */
inline void
update_latencies(const struct rpc_latencies *l, method_id_t id)
{
	// Trace values.
	trace[num_calls_made].id = id;
	trace[num_calls_made].bytes = l->len;
#if defined(TIMING_NATIVE)
	trace[num_calls_made].lat = l->exec.call; // native has no handoff with sink
	trace[num_calls_made].nvexec = l->exec.call;
#else
	trace[num_calls_made].lat = (l->lib.setup + l->lib.wait); // total cost
	trace[num_calls_made].nvexec = l->remote.batch_exec;
#endif

#if !defined(TIMING_NATIVE)
	// Aggregate values across the cluster.
    lat.lib.setup          +=  l->lib.setup;
    lat.lib.wait           +=  l->lib.wait;
    lat.exec.setup         +=  l->exec.setup;
    lat.rpc.append         +=  l->rpc.append;
    lat.rpc.send           +=  l->rpc.send;
    lat.rpc.wait           +=  l->rpc.wait;
    lat.rpc.recv           +=  l->rpc.recv;
    lat.remote.batch_exec  +=  l->remote.batch_exec;
#endif
	lat.exec.call          +=  l->exec.call;

	// Update counter
	num_calls_made++;
	if (unlikely(num_calls_made >= TRACE_MAX_CALLS)) {
		fprintf(stderr, "Error: not enough storage for trace. Reached %lu\n",
				TRACE_MAX_CALLS);
		exit(1);
	}
}

#define SYNC2STR(is_sync)	((is_sync) ? "sync" : "async")

inline void
print_latencies(void)
{
	unsigned long c = 0UL;

	printf(TIMERMSG_PREFIX "calls %lu\n", num_calls_made);

#if defined(TIMING) && defined(TIMING_NATIVE)
	printf(TIMERMSG_PREFIX "exec.call %lu\n", lat.exec.call);
#else
	printf(TIMERMSG_PREFIX
			"attach %lu detach %lu "
			"lib.setup %lu lib.wait %lu "
			"exec.setup %lu exec.call %lu "
			"rpc.append %lu rpc.send %lu rpc.wait %lu rpc.recv %lu "
			"remote.batch_exec %lu"
			"\n",
            attach, detach,
			lat.lib.setup, lat.lib.wait,
			lat.exec.setup, lat.exec.call,
			lat.rpc.append, lat.rpc.send, lat.rpc.wait, lat.rpc.recv,
			lat.remote.batch_exec);
#endif

    // c used here as call number; latencies are whatever include/util/timer.h
    // has configured, otherwise they are as specified in this line
	printf(TIMERMSG_PREFIX " -- BEGIN TRACE DUMP -- num id name size lat sync time_ns execlat\n");
	while (c < num_calls_made) {
		api_counts[trace[c].id]++;
		api_lat[trace[c].id] += trace[c].lat;
		printf(TIMERMSG_PREFIX
				"\t%lu %u %s %lu %lu %s %lu %lu\n",
				c, trace[c].id, method2str(trace[c].id), trace[c].bytes, trace[c].lat,
				SYNC2STR(method_synctable[trace[c].id]),
                (trace[c].ts.tv_sec * 1000000000UL + trace[c].ts.tv_nsec), trace[c].nvexec);
		c++;
	}
	printf(TIMERMSG_PREFIX " -- END TRACE DUMP --\n");

	// c used here as method_id
	printf(TIMERMSG_PREFIX " -- BEGIN CALL COUNTS -- count id name lat sync\n");
	for (c = (CUDA_INVALID_METHOD + 1); c < CUDA_METHOD_LIMIT; c++)
		if (api_counts[c])
			printf(TIMERMSG_PREFIX
					"\t%lu %lu %s %lu %s\n",
					api_counts[c], c, method2str(c), api_lat[c], SYNC2STR(method_synctable[c]));
	printf(TIMERMSG_PREFIX " -- END CALL COUNTS --\n");
}

#endif	/* TIMING */

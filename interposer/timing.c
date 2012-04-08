/**
 * @file timing.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 2, 2012
 * @brief Definitions and functions for timing the CUDA API.
 */

#include <util/timer.h>
#include "timing.h"

#ifdef TIMING

/*-------------------------------------- INTERNAL STATE ----------------------*/

// Timer resolution set by TIMER_ACCURACY in util/timer.h

static struct timer attach_timer; //! Timer used to measure joining the runtime
static uint64_t attach = 0UL; //! Time spent joining the runtime

static struct timer detach_timer; //! Timer used to measure departing the runtime
static uint64_t detach = 0UL; //! Time spent departing the runtime

static struct rpc_latencies lat; //! Aggregate latencies over all RPCs

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
update_latencies(const struct rpc_latencies *l)
{
	lat.lib.setup += l->lib.setup;
	lat.lib.wait += l->lib.wait;

	lat.exec.setup += l->exec.setup;
	lat.exec.call += l->exec.call;

	lat.rpc.append += l->rpc.append;
	lat.rpc.send += l->rpc.send;
	lat.rpc.wait += l->rpc.wait;
	lat.rpc.recv += l->rpc.recv;

	lat.remote.batch_exec += l->remote.batch_exec;
}

inline void
print_latencies(void)
{
	printf(TIMERMSG_PREFIX
			"lib.setup %lu lib.wait %lu "
			"exec.setup %lu exec.call %lu "
			"rpc.append %lu rpc.send %lu rpc.wait %lu rpc.recv %lu "
			"remote.batch_exec %lu"
			"\n",
			lat.lib.setup, lat.lib.wait,
			lat.exec.setup, lat.exec.call,
			lat.rpc.append, lat.rpc.send, lat.rpc.wait, lat.rpc.recv,
			lat.remote.batch_exec);
}

#endif	/* TIMING */

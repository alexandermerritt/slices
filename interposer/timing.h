/**
 * @file timing.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date Apr 2, 2012
 * @brief Definitions and functions for timing the CUDA API.
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <cuda/packet.h>
#include <util/timer.h>

#ifdef TIMING

struct call
{
	method_id_t id;
	unsigned long lat; // ency
	size_t bytes; // cuda_packet.len
};

void timers_init(void);
void update_latencies(const struct rpc_latencies *l, method_id_t id, size_t call_size);
void print_latencies(void);

void timers_start_attach(void);
void timers_start_detach(void);
void timers_stop_attach(void);
void timers_stop_detach(void);

#else	/* !TIMING */

#define timers_init()
#define update_latencies(lat,id,sz)
#define print_latencies()

#define timers_start_attach
#define timers_start_detach
#define timers_stop_attach
#define timers_stop_detach

#endif	/* TIMING */

#endif	/* TIMING_H_ */

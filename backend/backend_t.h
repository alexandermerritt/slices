/**
 * @file backend_t.h
 *
 * @date Sep 26, 2011
 * @author Magda Slawinska, aka Magic Magg, magg dot gatech at gmail dot edu
 *
 * This is the types that are used by the backend
 */

#ifndef BACKEND_T_H_
#define BACKEND_T_H_

/**
 * the structure about the resource the backend is running on;
 * it contains information about what the backend figured out about its own
 * execution environment: how many gpus, and cpus it posses, what is the
 * hostname, etc. This data are intended to be sent over the network to VC_Manager
 * by the BackendAgent;
 */
struct backend_rec {
	int 	cpu_count;
	int 	gpu_count;
	char*	hostname;
};

#endif /* BACKEND_T_H_ */

/**
 * @file libregistration.h
 *
 * @date October 2, 2011
 * @author Alex Merritt, merritt.alex@gatech.edu
 *
 * @brief Code herein defines the protocol used between an interposing library
 * and the backend process when initializing state to share memory for passing
 * marshalled CUDA function calls. Both the interposing library code and backend
 * code shall include this header.
 */

#ifndef __LIB_REGISTRATION_H
#define __LIB_REGISTRATION_H

/*-------------------------------------- DEFINITIONS -------------------------*/

typedef int regid_t;

/**
 * Event type passed to the callback function when an operation on a
 * registration has occured.
 */
enum callback_event {
	CALLBACK_NEW = 0,	//! Indicates a new registration has been created
	CALLBACK_DEL		//! Indicates a registration has been deleted
};

/*-------------------------------------- LIBRARY INTERFACE -------------------*/

int		reg_lib_init(void);
int		reg_lib_shutdown(void);
regid_t reg_lib_connect(void);
int		reg_lib_disconnect(regid_t id);
void*	reg_lib_get_shm(regid_t id);
size_t	reg_lib_get_shm_size(regid_t id);
// @todo TODO support growing the shm

/*-------------------------------------- BACKEND INTERFACE -------------------*/

int reg_be_init(unsigned int max_regs);
int reg_be_shutdown(void);
int reg_be_callback(void (*callback)(enum callback_event e, regid_t id));
void* reg_be_get_shm(regid_t id);
size_t reg_be_get_shm_size(regid_t id);

#endif

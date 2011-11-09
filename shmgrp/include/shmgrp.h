/**
 * @file shmgrp.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief TODO
 *
 * TODO LIST ASSUMPTIONS, LIMITATIONS, RESTRICTIONS HERE.
 */

#ifndef _SHMGRP_H
#define _SHMGRP_H

#include <unistd.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

//! This defines the maximum length usable for a group key. It must remain
//! short, as it is directly used in many places within the library where other
//! character limitations are in place.
#define SHMGRP_MAX_KEY_LEN	128

typedef int shmgrp_region_id;

struct shm_region
{
	size_t size;
	void *addr;
	shmgrp_region_id id;
};

typedef enum { MEMBERSHIP_JOIN = 1, MEMBERSHIP_LEAVE } group_event;
typedef enum { SHM_CREATE_REGION = 1, SHM_REMOVE_REGION } shm_event;

/**
 * TODO
 * maybe add a third arg identifying the group id? or allow a user-defined value
 * to be passed in?
 *
 * If one function is registered for multiple groups, it will need to be
 * thread-safe as multiple threads in the library will be calling into it.
 */
typedef void (*group_callback)(group_event, pid_t);

typedef void (*shm_callback)(shm_event, pid_t, shmgrp_region_id);

/*-------------------------------------- COMMON FUNCTIONS --------------------*/

/**
 * TODO
 */
int shmgrp_init(void);

/**
 * TODO
 */
int shmgrp_tini(void);

/**
 * TODO
 */
void* shmgrp_addr(shmgrp_region_id id);

/**
 * TODO
 */
const char * shmgrp_memb_str(group_event e);

/*-------------------------------------- LEADER FUNCTIONS --------------------*/

/**
 * TODO
 */
int shmgrp_open(const char *key, group_callback func);

/**
 * TODO
 */
int shmgrp_close(const char *key);

/**
 * This must be called with each invocation of the callback registered with the
 * creation of a group. It finalizes the addition of a new member to the group,
 * identified by the PID passed as an argument to the callback (new members will
 * spin indefinitely inside join() if this is not called).
 *
 * The callback will be invoked every time a member process requests to create
 * or destroy a shared memory segment. If the same function is registered more
 * than once, it will need to be thread-safe.
 */
int shmgrp_establish_member(const char *key, pid_t pid, shm_callback func);

/*-------------------------------------- MEMBER FUNCTIONS --------------------*/

/**
 * TODO
 */
int shmgrp_join(const char *key);

/**
 * TODO
 */
int shmgrp_leave(const char *key);

/**
 * TODO
 * add some size argument
 *
 * the value of the ID returned is common between both the member and the
 * leader, but is not unique across members of the same group, nor across
 * groups. in other words, if you were to communicate this ID out-of-band to the
 * leader or member involved, they would be able to identify the same physical
 * region of memory.
 */
int shmgrp_mkreg(const char *key, size_t size, shmgrp_region_id *id);

/**
 * TODO
 */
int shmgrp_rmreg(const char *key, shmgrp_region_id id);

#endif /* _SHMGRP_H */

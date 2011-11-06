/**
 * @file shmgrp.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief TODO
 */

#ifndef _SHMGRP_H
#define _SHMGRP_H

#include <unistd.h>

#define SHMGRP_MAX_KEY_LEN	128

/**
 * TODO
 */
int shmgrp_init(void);

/**
 * TODO
 */
int shmgrp_tini(void);

typedef enum {MEMBERSHIP_JOIN, MEMBERSHIP_LEAVE} membership_event;

/**
 * TODO
 * maybe add a third arg identifying the group id? or allow a user-defined value
 * to be passed in?
 */
typedef int (*membership_callback)(membership_event, pid_t);

/**
 * TODO
 * key must begin with /
 */
int shmgrp_open(const char *key, membership_callback func);

/**
 * TODO
 */
int shmgrp_close(/*TODO*/);

#if 0
/**
 * TODO
 */
int shmgrp_join(/*TODO*/);

/**
 * TODO
 */
int shmgrp_leave(/*TODO*/);

/**
 * TODO
 * add some size argument
 */
int shmgrp_mkreg(/*TODO*/);

/**
 * TODO
 */
int shmgrp_rmreg(/*TODO*/);
#endif

#endif /* _SHMGRP_H */

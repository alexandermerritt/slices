/**
 * @file common.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief TODO
 */

#include <shmgrp.h>

extern int shmgrp_init_leader(void);
extern int shmgrp_tini_leader(void);
extern int shmgrp_init_member(void);
extern int shmgrp_tini_member(void);

/*-------------------------------------- COMMON PUBLIC INTERFACE -------------*/

int shmgrp_init(void)
{
	// Initialize both states as a process may want to simultaneously lead some
	// groups while be a member of others.
	int err;
	err = shmgrp_init_leader();
	if (err < 0)
		return err;
	err = shmgrp_init_member();
	if (err < 0)
		return err;
	return 0;
}

int shmgrp_tini(void)
{
	int err;
	err = shmgrp_tini_leader();
	if (err < 0)
		return err;
	err = shmgrp_tini_member();
	if (err < 0)
		return err;
	return 0;
}

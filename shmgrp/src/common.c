/**
 * @file common.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief TODO
 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include <shmgrp.h>

#include "common.h"

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

const char * shmgrp_memb_str(group_event e)
{
	switch (e) {
		case MEMBERSHIP_JOIN:
			return "Membership Join";
		case MEMBERSHIP_LEAVE:
			return "Membership Leave";
		default:
			return "Uknown Membership Event";
	}
}

/*-------------------------------------- INTERNAL UTILITIES ------------------*/

bool verify_userkey(const char *key)
{
	// Key cannot contain a slash character anywhere
	if (!key || strchr(key, '/') != NULL)
		return false;
	// And must conform to the required length
	if (strlen(key) >= SHMGRP_MAX_KEY_LEN)
		return false;
	return true;
}

int group_dir_exists(const char *key, bool *exists)
{
	int err;
	struct stat statbuf;
	char memb_dir[MAX_LEN];

	*exists = false;
	memset(&statbuf, 0, sizeof(statbuf));
	memset(memb_dir, 0, MAX_LEN);

	snprintf(memb_dir, MAX_LEN, MEMB_DIR_FMT, key);
	err = stat(memb_dir, &statbuf);
	if (err < 0 && errno != ENOENT) // ENOENT means it doesn't exist
		return -(errno);
	*exists = true;
	return 0;
}

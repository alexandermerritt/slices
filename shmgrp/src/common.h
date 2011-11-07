/**
 * @file common.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief This file contains some standard definitions across all files
 * implementing shmgrp.
 */

#ifndef _COMMON_H
#define _COMMON_H

#include <dirent.h>
#include <stdbool.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

//! Longest length for any key, be it an inotify file, group directory path, or
//! shared memory region key. This length may include the user-supplied group
//! key.
#define MAX_LEN				255

/*
 * Groups have their own membership directory. Members who join a group create a
 * file in this directory. The name of this file need not be unique across
 * groups, as these files are separated into different directories. All of this
 * is used with the inotify interface provided by Linux.
 *
 * In the filesystem:
 * 		/   dir prefix  / group key / pid			Naming convention
 *       Directory Path   Directory   File			Structure
 * where 'pid' is the name of a file a process creates from its own PID (the
 * shmgrp library peforms this) to communicate the desire to join the group
 * 'group key'.
 *
 * inotify is used here to observe for group arrivals and departures.
 */

#define MEMB_DIR_PREFIX_DIR		"/tmp/shmgrp/"
#define MEMB_DIR_FMT			MEMB_DIR_PREFIX_DIR "%s/" // group key
#define MEMB_DIR_PERMS			0770
#define MEMB_FILE_FMT			MEMB_DIR_FMT "%d"		// pid
#define MEMB_FILE_PERMS			0660

/*
 * Message queues exist once for each pairing between a process and a group
 * leader.  Their names (keys) must be unique across the entire system. The
 * POSIX message queue interface requires keys to consist of no more than
 * NAME_MAX characters, where the first must be a slash, and no other character
 * may be a slash.
 *
 * man 7 mq_overview
 *
 * According to Linux, shmgrp message queues have the following naming
 * convention:
 * 		mq prefix - group key - pid
 * where 'pid' is the PID of the member process.
 *
 * Message queues are used here as a medium for carrying messages between an
 * existing member and a leader. One use is to establish or destroy shared
 * memory segments.
 */

#define MQ_PREFIX				"/shmgrp-"
#define MQ_NAME_FMT				MQ_PREFIX "%s-%s"	// group key - pid
#define MQ_PERMS				0660
#define MQ_OPEN_LEADER_FLAGS	(O_RDWR | O_CREAT | O_EXCL)
#define MQ_OPEN_MEMBER_FLAGS	(O_RDWR)
#define MQ_ID_INVALID_VALUE		((mqd_t) - 1) // RTFM
#define MQ_ID_IS_VALID(m)		((m) != MQ_ID_INVALID_VALUE)

//! Maximum number of messages allowed to queue up.
#define MQ_MAX_MESSAGES			8

/*
 * Shared memory files exist with any number of instances between each leader
 * and member within a group. SHM key names have the same requirements as
 * message queue keys (see above). SHM keys must be unique across the entire
 * system, as they also are a global resource.
 *
 * man 7 shm_overview
 *
 * According to Linux, shmgrp regions have the following naming convention:
 *		shm prefix - group key - pid - region ID
 * where, again, 'pid' is the PID of the member process. 'region ID' is a
 * monotonically increasing identifier supplied by the shmgrp library.
 */

#define SHM_PREFIX				"/shmgrp-"
#define SHM_NAME_FMT			SHM_PREFIX "%s-%s-%s" // group key - pid - reg
#define SHM_PERMS				0660
#define SHM_LEADER_OPEN_FLAGS	(O_RDWR | O_CREAT | O_EXCL)
#define SHM_MEMBER_OPEN_FLAGS	(O_RDWR)

/*-------------------------------------- STRUCTURES --------------------------*/

typedef enum {MESSAGE_CREATE_SHM, MESSAGE_REMOVE_SHM} message_type;

struct message
{
	message_type type;
};

/*-------------------------------------- FUNCTIONS ---------------------------*/

bool verify_userkey(const char *key);
int group_dir_exists(const char *key, bool *exists);

#endif

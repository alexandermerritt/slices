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
#define MEMB_DIR_PERMS			0777
#define MEMB_FILE_FMT			MEMB_DIR_FMT "%d"		// pid
#define MEMB_FILE_PERMS			0666

/*
 * Message queues exist once for each pairing between a process and a group
 * leader.  Their names (keys) must be unique across the entire system. The
 * POSIX message queue interface requires keys to consist of no more than
 * NAME_MAX characters, where the first must be a slash, and no other character
 * may be a slash.
 *
 * man 7 mq_overview
 *
 * shmgrp message queues have the following naming convention in this library:
 * 		mq prefix - group key - pid
 * where 'pid' is the PID of the member process.
 *
 * Message queues are used here as a medium for carrying messages between an
 * existing member and a leader. One use is to establish or destroy shared
 * memory segments.
 *
 * An important point to note: if a process writes to a message queue and
 * immediately reads, it will get its own message back out. MQs are handled as
 * common-access queues by any and all processes which have opened it. That is
 * why two types exist between each member process and a leader for each group:
 * one for each direction of communication. Each of the two uni-direcitonal MQs
 * with format descriptors below has 'leader' or 'member' in it signifying who
 * owns that MQ and the direction messages flow along it: the owner is
 * responsible for creating and destroying it, and the owner is the source for
 * messages on its own MQ (meaning it performs send on its MQ and reads on
 * the other).
 */

#define MQ_PREFIX				"/shmgrp"
#define MQ_NAME_FMT_BASE		MQ_PREFIX "-%s-%d"	// group key - pid
#define MQ_NAME_FMT_LEADER		MQ_NAME_FMT_BASE "-leader"
#define MQ_NAME_FMT_MEMBER		MQ_NAME_FMT_BASE "-member"

#define MQ_PERMS				(S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | \
														S_IROTH | S_IWOTH)

//! Flags used to create the MQ under ownership. A leader will use these flags
//! to open MQ_NAME_FMT_LEADER. Note that mq_open takes four arguments when
//! passed the oflags bit O_CREAT: if you ever see an EFAULT errno value when
//! mq_open fails, it is probably because you are passing only two.
#define MQ_OPEN_OWNER_FLAGS		(O_RDWR | O_CREAT | O_EXCL | O_NONBLOCK)

//! Flags used to connect to an MQ owned by another entity. A leader will use
//! these flags to open MQ_NAME_FMT_MEMBER. Note that mq_open takes two
//! arguments when not passed the oflags bit O_CREAT.
#define MQ_OPEN_CONNECT_FLAGS	(O_RDWR)

#define MQ_ID_INVALID_VALUE		((mqd_t) - 1) // RTFM
#define MQ_ID_IS_VALID(m)		((m) != MQ_ID_INVALID_VALUE)

//! Maximum number of messages allowed to queue up.
#define MQ_MAX_MESSAGES			8

//! Maximum message size allowable. Just set to the size of our structure.
#define MQ_MAX_MSG_SIZE			(sizeof(struct message))

//! Default priority for messages
#define MQ_DFT_PRIO				0

/*
 * Shared memory files exist with any number of instances between each leader
 * and member within a group. SHM key names have the same requirements as
 * message queue keys (see above). SHM keys must be unique across the entire
 * system, as they also are a global resource.
 *
 * man 7 shm_overview
 *
 * shmgrp regions have the following naming convention in this library:
 *		shm prefix - group key - pid - region ID
 * where, again, 'pid' is the PID of the member process. 'region ID' is a
 * monotonically increasing identifier supplied by the shmgrp library.
 */

#define SHM_PREFIX				"/shmgrp"
#define SHM_NAME_FMT			SHM_PREFIX "-%s-%d-%d" // grp key - pid - regid
#define SHM_PERMS				0666
#define SHM_OPEN_LEADER_FLAGS	(O_RDWR | O_CREAT | O_EXCL)
#define SHM_OPEN_MEMBER_FLAGS	(O_RDWR)

#define MMAP_PERMS				(PROT_READ | PROT_WRITE)
#define MMAP_FLAGS				(MAP_SHARED)

/*-------------------------------------- STRUCTURES --------------------------*/

typedef enum { MESSAGE_CREATE_SHM = 1, MESSAGE_REMOVE_SHM } message_type;

/**
 * A struct describing the data portion of request/response messages for
 * creating and destroying shared memory mappings.
 *
 * Requesting a new region:
 * 		A CREATE request is sent by the member, setting the size and ignoring
 * 		both the ID and status fields.  The leader receives this, creates the
 * 		mapping on its side, then replies with the same message, but filling in
 * 		the ID to be a value of its choosing. Once the response is received from
 * 		the leader the member examines the status field to see if the request
 * 		was accepted (or if there was an error processing the request), the
 * 		member then creates internal region state and maps in the region.
 *
 * Closing an existing region:
 * 		The member first deallocates and unmaps the region state, then issues a
 * 		REMOVE request to the leader, setting the region ID to the value
 * 		returned by the leader on a prior CREATE. The size and status fields are
 * 		ignored. The leader receives this, looks up the ID, deallocates and
 * 		unmaps the region on its end, before sending a response to the member
 * 		that it has finished. The leader fills in the status field to indicate
 * 		whether or not it was successful with an errno code.
 */
struct region_descriptor
{
	size_t size;
	shmgrp_region_id id;
	int status; //! Contains zero, or a negative errno code.
};

// TODO Add other messages in the future, if the need arises. Each message type
// should get its own set of functions to expose the functionality to a caller.

struct message
{
	message_type type;
	union {
		struct region_descriptor region;
	} m; // m for message
};

/*-------------------------------------- FUNCTIONS ---------------------------*/

bool verify_userkey(const char *key);
int group_dir_exists(const char *key, bool *exists);

int shmgrp_init_leader(void);
int shmgrp_tini_leader(void);
int shmgrp_init_member(void);
int shmgrp_tini_member(void);

#endif

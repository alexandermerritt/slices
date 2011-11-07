/**
 * @file shmgrp.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-05
 * @brief This file manages state associated with members of groups.
 */

#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <shmgrp.h>

#include "list.h"
#include "common.h"

/*-------------------------------------- INTERNAL DEFINITIONS ----------------*/

/*-------------------------------------- INTERNAL TYPES ----------------------*/

struct region
{
	struct list_head link;
	int fd;
	void *addr;
	size_t size;
};

struct membership
{
	struct list_head link;
	struct list_head region_list;
	pid_t pid;
	mqd_t mq_id;
	char mq_name[MAX_LEN];
	char user_key[MAX_LEN];
};

struct memberships
{
	struct list_head list; //! List of memberships
	pthread_mutex_t lock;
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct memberships *memberships = NULL;

/*-------------------------------------- INTERNAL STATE OPS ------------------*/

/*
 * Membership operations.
 */

#define membership_for_each_region(membership, region)	\
	list_for_each_entry(region, &((membership)->region_list), link)

static inline bool
membership_has_regions(struct membership *membership)
{
	return !list_empty(&(membership->region_list));
}

/*
 * Membership list operations.
 */

#define memberships_for_each_membership(memberships, membership)	\
	list_for_each_entry(membership, &((memberships)->list), link)

static inline void
memberships_add_membership(struct memberships *memberships,
		struct membership *membership)
{
	list_add(&(membership->link), &(memberships->list));
}

static inline struct membership *
memberships_get_membership(struct memberships *memberships, const char *key)
{
	struct membership *membership;
	memberships_for_each_membership(memberships, membership)
		if (strcmp(key, membership->user_key) == 0)
			return membership;
	return NULL;
}

static inline void
__memberships_rm_membership(struct membership *membership)
{
	list_del(&(membership->link));
}

static inline void
memberships_rm_membership(struct memberships *memberships, const char *key)
{
	struct membership *membership;
	membership = memberships_get_membership(memberships, key);
	if (membership)
		__memberships_rm_membership(membership);
}

/*-------------------------------------- PUBLIC INTERFACE --------------------*/

int shmgrp_init_member(void)
{
	memberships = calloc(1, sizeof(*memberships));
	if (!memberships)
		return -ENOMEM;
	INIT_LIST_HEAD(&memberships->list);
	pthread_mutex_init(&memberships->lock, NULL);
	return 0;
}

int shmgrp_tini_member(void)
{
	// FIXME clear other resources within each existing group
	if (memberships)
		free(memberships);
	memberships = NULL;
	return 0;
}

int shmgrp_join(const char *key)
{
	struct membership *new_memb = NULL;
	int err, exit_errno;
	bool memb_dir_exists;
	char memb_file[MAX_LEN];
	char pid_str[MAX_LEN];

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	memset(pid_str, 0, MAX_LEN);
	memset(memb_file, 0, MAX_LEN);

	// First see if that group even exists. Read the filesystem to see if the
	// group registration directory has been created for this key.
	err = group_dir_exists(key, &memb_dir_exists);
	if (err < 0) {
		exit_errno = -(err);
		goto fail;
	}
	if (!memb_dir_exists) {
		exit_errno = -ENOENT;
		goto fail;
	}

	// Lock the membership list. Then we can see if we already have a membership
	// with that key before proceeding. If none exists, hold lock while we
	// initialize a new member.
	pthread_mutex_lock(&(memberships->lock));
	new_memb = memberships_get_membership(memberships, key);
	if (new_memb) {
		pthread_mutex_unlock(&(memberships->lock));
		exit_errno = -EEXIST;
		goto fail;
	}

	new_memb = calloc(1, sizeof(*new_memb));
	if (!new_memb) {
		exit_errno = -ENOMEM;
		goto fail;
	}
	INIT_LIST_HEAD(&new_memb->link);
	INIT_LIST_HEAD(&new_memb->region_list);
	new_memb->pid = getpid();
	strncpy(new_memb->user_key, key, MAX_LEN);

	// Write our file into the membership directory, this will cause the leader
	// to create a message queue for us.
	snprintf(memb_file, MAX_LEN, MEMB_FILE_FMT,
			new_memb->user_key, new_memb->pid);
	err = creat(memb_file, MEMB_FILE_PERMS);
	if (err < 0) {
		pthread_mutex_unlock(&(memberships->lock));
		exit_errno = -(errno);
		goto fail;
	}

	// Open the message queue. If it doesn't exist, spin until it does.
	// FIXME We don't want to spin, because a malicious/buggy leader might never
	// open a queue.
	new_memb->mq_id = MQ_ID_INVALID_VALUE;
	snprintf(pid_str, MAX_LEN, "%d", new_memb->pid);
	snprintf(new_memb->mq_name, MAX_LEN, MQ_NAME_FMT, key, pid_str);
	do { // TODO clean up this loop, it's ugly
		new_memb->mq_id = mq_open(new_memb->mq_name, MQ_OPEN_MEMBER_FLAGS);
		if (!MQ_ID_IS_VALID(new_memb->mq_id)) {
			if (errno == ENOENT) {
				continue;
			} else {
				pthread_mutex_unlock(&memberships->lock);
				exit_errno = -(errno);
				goto fail;
			}
		}
		break; // opened successfully, leave loop
	} while(1);

	// Add the member as the very last operations (else we might have to remove
	// it again in the fail code below).
	memberships_add_membership(memberships, new_memb);
	pthread_mutex_unlock(&(memberships->lock));
	return 0;

fail:
	// lock should not be held here
	if (new_memb)
		free(new_memb);
	return exit_errno;
}

int shmgrp_leave(const char *key)
{
	int err, exit_errno;
	struct membership *membership = NULL;
	char memb_file[MAX_LEN];

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	memset(memb_file, 0, MAX_LEN);

	// Lock the list. See if a group exists with that key. If so, remove it from
	// the list.
	pthread_mutex_lock(&memberships->lock);
	membership = memberships_get_membership(memberships, key);
	if (!membership) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	__memberships_rm_membership(membership);

	// Now we are free to operate on the membership without further locking.
	if (membership_has_regions(membership)) {
		// What TODO?
		// deallocate/close all regions
	}

	// Remove our membership registration file. This should be done AFTER
	// closing all the memory regions, as that will tell the leader to do the
	// same before it finds us leaving the group.
	snprintf(memb_file, MAX_LEN, MEMB_FILE_FMT,
			membership->user_key, membership->pid);
	err = unlink(memb_file);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -(errno);
		goto fail;
	}

	pthread_mutex_unlock(&memberships->lock);
	return 0;

fail:
	return exit_errno;
}

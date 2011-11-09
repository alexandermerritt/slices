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
#include <sys/mman.h>
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
	char shm_file[MAX_LEN];
	shmgrp_region_id id; // the value of id is created by the leader
};

struct membership
{
	struct list_head link;
	struct list_head region_list;
	pid_t pid;
	mqd_t mq_in;
	mqd_t mq_out;
	char mqname_in[MAX_LEN];
	char mqname_out[MAX_LEN];
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
 * Region operations.
 */


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

static inline struct region *
membership_get_region(struct membership *membership, shmgrp_region_id id)
{
	struct region *region;
	membership_for_each_region(membership, region)
		if (region->id == id)
			return region;
	return NULL;
}

static inline void
__membership_rm_region(struct region *region)
{
	list_del(&region->link);
}

static inline void
membership_rm_region(struct membership *membership, shmgrp_region_id id)
{
	struct region *region;
	region = membership_get_region(membership, id);
	if (region)
		__membership_rm_region(region);
}

static inline void
membership_add_region(struct membership *membership, struct region *region)
{
	list_add(&region->link, &membership->region_list);
}

// Send a message to the membership's message queue
// If the message queue is full
// 		and was created as blocking, this will block
// 		and was created as nonblocking, this will spin
// until the message can be inserted
static int
membership_send_message(struct membership *memb, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_send(memb->mq_out, (char*)msg, sizeof(*msg), MQ_DFT_PRIO);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
		if (errno == EAGAIN)
			goto again; // mq created as nonblock & is full, try again
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

// Receive a message from the membership's message queue
// If the message queue is empty
// 		and was created as blocking, this will block
// 		and was created as nonblocking, this will spin
// until a message arrives
static int
membership_recv_message(struct membership *memb, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_receive(memb->mq_in, (char*)msg, sizeof(*msg), NULL);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
		if (errno == EAGAIN)
			goto again; // mq created as nonblock & is empty, try again
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
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
	struct mq_attr qattr;

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	memset(pid_str, 0, MAX_LEN);
	memset(memb_file, 0, MAX_LEN);
	memset(&qattr, 0, sizeof(qattr));

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
	new_memb->mq_in = MQ_ID_INVALID_VALUE;
	new_memb->mq_out = MQ_ID_INVALID_VALUE;
	new_memb->pid = getpid();
	strncpy(new_memb->user_key, key, MAX_LEN);

	// Create the MQ under our ownership. Owned MQs are used for sending. This
	// must be created BEFORE notifying the leader (creat triggering inotify).
	// The leader will perform a non-create open of our MQ, and if that fails,
	// the leader will not be able to join us. The expectation is that the
	// existance of our owned MQ confirms our request to join via inotify.
	snprintf(new_memb->mqname_out,
			MAX_LEN, MQ_NAME_FMT_MEMBER,
			key, new_memb->pid);
	qattr.mq_maxmsg = MQ_MAX_MESSAGES;
	qattr.mq_msgsize = MQ_MAX_MSG_SIZE;
	new_memb->mq_out = mq_open(new_memb->mqname_out,
			MQ_OPEN_OWNER_FLAGS, MQ_PERMS, &qattr);
	if (!MQ_ID_IS_VALID(new_memb->mq_out)) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -(errno);
		goto fail;
	}

	// Write our file into the membership directory, this will cause the leader
	// to create a message queue for us. It will attempt to open the MQ under
	// our ownership to verify the join request, then open the MQ under its
	// ownership, which we open below.
	snprintf(memb_file, MAX_LEN, MEMB_FILE_FMT,
			new_memb->user_key, new_memb->pid);
	err = creat(memb_file, MEMB_FILE_PERMS);
	if (err < 0) {
		pthread_mutex_unlock(&(memberships->lock));
		exit_errno = -(errno);
		goto fail;
	}

	// Open the MQ under ownership of the leader. This MQ is used for receiving.
	// Spin until we are able to open it (that is, until the leader creates it).
	new_memb->mq_in = MQ_ID_INVALID_VALUE;
	snprintf(new_memb->mqname_in, MAX_LEN, MQ_NAME_FMT_LEADER, key, new_memb->pid);
	do {
		new_memb->mq_in =
			mq_open(new_memb->mqname_in, MQ_OPEN_CONNECT_FLAGS);
		if (!MQ_ID_IS_VALID(new_memb->mq_in)) {
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
	if (new_memb) {
		if (MQ_ID_IS_VALID(new_memb->mq_in)) {
			mq_close(new_memb->mq_in);
		}
		if (MQ_ID_IS_VALID(new_memb->mq_out)) {
			mq_close(new_memb->mq_out);
			mq_unlink(new_memb->mqname_out);
		}
		free(new_memb);
	}
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

	if (membership_has_regions(membership)) {
		// What TODO?
		// deallocate/close all regions
	}

	// Remove our membership registration file. This should be done AFTER
	// closing all the memory regions, as the leader will delete the shm object
	// files once notified.
	snprintf(memb_file, MAX_LEN, MEMB_FILE_FMT,
			membership->user_key, membership->pid);
	err = unlink(memb_file);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -(errno);
		goto fail;
	}

	// Close both MQs.
	err = mq_close(membership->mq_in);
	if (err < 0)
		fprintf(stderr, "Could not close MQ-in for membership %s\n",
				membership->user_key);
	err = mq_close(membership->mq_out);
	if (err < 0)
		fprintf(stderr, "Could not close MQ-out for membership %s\n",
				membership->user_key);
	err = mq_unlink(membership->mqname_out);
	if (err < 0)
		fprintf(stderr, "Could not unlink MQ-%s for membership %s\n",
				membership->mqname_out, membership->user_key);

	free(membership);
	pthread_mutex_unlock(&memberships->lock);
	return 0;

fail:
	return exit_errno;
}

int shmgrp_mkreg(const char *key, size_t size, shmgrp_region_id *id)
{
	int err, exit_errno = 0;
	struct membership *memb = NULL;
	struct region *region = NULL;
	struct message message;

	if (!verify_userkey(key) || size == 0 || !id) {
		exit_errno = -EINVAL;
		goto fail;
	}

	// Need to be thread-safe
	pthread_mutex_lock(&memberships->lock);

	// Locate group associated with the key
	memb = memberships_get_membership(memberships, key);
	if (!memb) {
		pthread_mutex_lock(&memberships->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// Create a memory region request message
	memset(&message, 0, sizeof(message));
	message.type = MESSAGE_CREATE_SHM;
	message.m.region.size = size;

	// Send it to the leader, wait for reply. If successful, the leader will
	// have created the shm object file before sending a reply.
	err = membership_send_message(memb, &message);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = err;
		goto fail;
	}
	memset(&message, 0, sizeof(message));
	err = membership_recv_message(memb, &message);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = err;
		goto fail;
	}
	// Check the status, whether or not we may proceed.
	if (message.m.region.status < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -EPERM; // DENIED!!
		goto fail;
	}

	// Create a region object
	region = calloc(1, sizeof(*region));
	if (!region) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -ENOMEM;
		goto fail;
	}
	INIT_LIST_HEAD(&region->link);
	region->size = size; // We assume the leader grants us what we ask
	region->id = message.m.region.id;
	region->addr = MAP_FAILED;
	region->fd = -1;

	// Map in the shared memory descriptor
	// TODO Maybe the leader can tell us the filename instead?
	snprintf(region->shm_file, MAX_LEN, SHM_NAME_FMT,
			memb->user_key, getpid(), region->id);
	region->fd = shm_open(region->shm_file, SHM_OPEN_MEMBER_FLAGS, SHM_PERMS);
	if (region->fd < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -(errno); // FUBAR
		goto fail;
	}
	region->addr = mmap(NULL, region->size,
			MMAP_PERMS, MMAP_FLAGS, region->fd, 0);
	if (region->addr == MAP_FAILED) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -(errno);
		goto fail;
	}

	*id = region->id;
	membership_add_region(memb, region);
	pthread_mutex_unlock(&memberships->lock);
	return 0;

fail:
	if (region) {
		if (region->addr != MAP_FAILED)
			munmap(region->addr, region->size);
		if (region->fd >= 0)
			close(region->fd);
		free(region);
	}
	return exit_errno;
}

int shmgrp_rmreg(const char *key, shmgrp_region_id id)
{
	int err, exit_errno = 0;
	struct membership *memb = NULL;
	struct region *region = NULL;
	struct message message;

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	// Need to be thread-safe
	pthread_mutex_lock(&memberships->lock);

	// Look up the membership and region
	memb = memberships_get_membership(memberships, key);
	if (!memb) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	region = membership_get_region(memb, id);
	if (!region) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	__membership_rm_region(region);

	// Close memory state
	err = munmap(region->addr, region->size); // FIXME check err
	err = close(region->fd); // FIXME check err

	// send message to leader, it will unlink the shm file
	memset(&message, 0, sizeof(message));
	message.type = MESSAGE_REMOVE_SHM;
	message.m.region.id = region->id;
	err = membership_send_message(memb, &message);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = err;
		goto fail;
	}
	memset(&message, 0, sizeof(message));
	err = membership_recv_message(memb, &message);
	if (err < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = err;
		goto fail;
	}
	// Check the status
	if (message.m.region.status < 0) {
		pthread_mutex_unlock(&memberships->lock);
		exit_errno = -EPERM; // DENIED!!
		goto fail;
	}

	free(region);
	pthread_mutex_unlock(&memberships->lock);
	return 0;

fail:
	return exit_errno;
}

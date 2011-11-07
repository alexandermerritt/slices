/**
 * @file leader.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-05
 * @brief This file manages state associated with leaders of groups.
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

#define INOTIFY_NUM_EVENTS	32
#define INOTIFY_EVENT_SZ	(sizeof(struct inotify_event))
#define INOTIFY_BUFFER_SZ	(INOTIFY_NUM_EVENTS * INOTIFY_EVENT_SZ)

/*-------------------------------------- INTERNAL TYPES ----------------------*/

struct region
{
	struct list_head link;
	int fd;
	void *addr;
	size_t size;
	char shm_key[MAX_LEN];
};

struct member
{
	struct list_head link;
	struct list_head region_list;
	pid_t pid;
	mqd_t mq_id;
	char mq_name[MAX_LEN];
};

// forward declaration
struct inotify_thread_state;

struct group
{
	struct list_head link;
	char user_key[SHMGRP_MAX_KEY_LEN];
	char memb_dir[MAX_LEN];
	// TODO other keys needed?
	struct list_head member_list;
	membership_callback notify;
	struct inotify_thread_state *inotify_thread;
	pthread_mutex_t lock;
};

struct groups
{
	struct list_head list; //! List of groups
	pthread_mutex_t lock;
};

/**
 * Protocol errors. These indicate events that occur which violate our
 * expectations regarding the use of the underlying system IPC, etc and are not
 * actual errors from improper use of system calls, etc. They're pretty much
 * impossible to encounter.
 *
 * These can happen, for example, if someone maliciously deletes the inotify
 * directory we watch without using the library.  This action triggers an
 * inotify event for a deletion event, but with a zero-length field (no filename
 * within the directory triggered it).
 */
typedef enum
{
	//! We only use one wd, thus a different value is unexpected
	PROTO_WRONG_WD = 1,

	//! Another condition we check for, but I dunno how to interpret it.
	PROTO_ZERO_READ_LEN,

	//! Indicates a 'nameless event'. Since we trigger ONLY on create/delete
	//! this can signify the directory itself was deleted or something else bad
	PROTO_ZERO_EVT_LEN,

	//! The current implementation only considers inotify triggers for file
	//! creation and deletion. Any other event would cause inotify to tell us
	//! with a different mask, such as opening or renaming a file. But we aren't
	//! considering them.
	PROTO_UNSUPPORTED_EVT_MASK

} proto_code;

/**
 * State associated with each inotify thread. An instance is created when a
 * leader chooses to create a group, and is passed as the argument to an inotify
 * thread we create to manage the group.
 *
 * We hijack the errno EPROTO to mean something has gone wrong with our method
 * of using inotify (-> our protocol). Thus proto_code is available to provide
 * more detail. In doing this, we assume no errno value returned will ever be
 * EPROTO unless it is explicitly set by us.
 */
struct inotify_thread_state
{
	// Thread-specific information
	bool is_alive;
	pthread_t tid;
	// Thread's exit reason
	// 0=okay, else < 0 indicating (-errno) value
	// if -EPROTO, read proto_error for specific detail
	int exit_code;
	proto_code proto_error;

	// Membership information
	struct group *group;

	// Inotify state for this thread
	int inotify_fd;
	char *events; //! Array of inotify events (not strings)
};

/*-------------------------------------- INTERNAL STATE ----------------------*/

static struct groups *groups = NULL;

/*-------------------------------------- INTERNAL STATE OPS ------------------*/

/*
 * No locking is performed within _most_ of the internal functions in this
 * section.  All locks that need to be held are assumed to have been acquired
 * for such functions. Functions which do acquire locks perform more complex
 * manipulation of state, such as creating and adding new members to groups.
 * Also, most error checking is avoided to keep the code simple, easy to read
 * and fast.
 */

/*
 * Memory operations.
 */

// TODO map, unmap, etc
// will also need some stuff for message queue mgmt

/*
 * Member operations.
 */

#define member_for_each_region(member, region)	\
	list_for_each_entry(region, &((member)->region_list), link)

static inline void
member_add_region(struct member *mem, struct region *reg)
{
	list_add(&(reg->link), &(mem->region_list));
}

#if 0
static inline int member_rm_region(struct member *mem, int reg_id)
{
	struct region *reg = NULL;
	if (!mem || reg_id < 0)
		return -1;
	member_for_each_region(member, reg)
		if (reg->id == reg_id)
			break;
	if (!reg || reg->id != reg_id)
		return -1;
	list_del(&(reg->link));
	return 0;
}
#endif

static inline bool
member_has_regions(struct member *mem)
{
	return !list_empty(&(mem->region_list));
}

/*
 * Group operations.
 */

#define group_for_each_member(group, member)	\
	list_for_each_entry(member, &((group)->member_list), link)

static inline void
group_add_member(struct group *group, struct member *mem)
{
	list_add(&(mem->link), &(group->member_list));
}

static inline bool
group_has_members(struct group *group)
{
	return !list_empty(&(group->member_list));
}

static struct member *
group_get_member(struct group *group, pid_t pid)
{
	struct member *member;
	group_for_each_member(group, member)
		if (member->pid == pid)
			break;
	if (!member || member->pid != pid)
		return NULL;
	return member;
}

static inline void
__group_rm_member(struct group *group, struct member *member)
{
	list_del(&member->link);
}

static inline int
group_rm_member(struct group *group, pid_t pid)
{
	struct member *member;
	member = group_get_member(group, pid);
	__group_rm_member(group, member);
	return 0;
}

static bool
group_has_member(struct group *group, pid_t pid)
{
	return (group_get_member(group, pid) != NULL);
}

static inline void
group_stop_inotify(struct group *group)
{
	if (group->inotify_thread->is_alive) {
		pthread_cancel(group->inotify_thread->tid);
		pthread_join(group->inotify_thread->tid, NULL);
	}
}

// forward declaration
static void process_message(union sigval);

static int
group_member_join(struct group *group, pid_t pid, struct member **new_memb)
{
	int err, exit_errno;
	struct mq_attr qattr;
	struct sigevent event;
	struct member *memb = NULL;
	char pid_str[MAX_LEN];

	memset(&event, 0, sizeof(event));
	memset(&qattr, 0, sizeof(qattr));
	memset(pid_str, 0, MAX_LEN);

	// Lock the group in case there are multiple joins to the same group at the
	// same time. Release lock when a new member has been added.
	pthread_mutex_lock(&group->lock);

	// Disallow joins for the same PID.
	if (group_has_member(group, pid)) {
		pthread_mutex_unlock(&group->lock);
		exit_errno = -EEXIST;
		goto fail;
	}

	memb = calloc(1, sizeof(*memb));
	if (!memb) {
		fprintf(stderr, "Out of memory\n");
		pthread_mutex_unlock(&group->lock);
		exit_errno = -ENOMEM;
		goto fail;
	}
	INIT_LIST_HEAD(&memb->link);
	INIT_LIST_HEAD(&memb->region_list);
	memb->pid = pid;
	memb->mq_id = MQ_ID_INVALID_VALUE;

	// Create a message queue and turn on asynchronous notification.
	snprintf(pid_str, MAX_LEN, "%d", pid);
	snprintf(memb->mq_name, MAX_LEN, MQ_NAME_FMT, group->user_key, pid_str);
	qattr.mq_maxmsg = MQ_MAX_MESSAGES;
	qattr.mq_msgsize = sizeof(struct message);
	memb->mq_id = mq_open(memb->mq_name, MQ_OPEN_LEADER_FLAGS, MQ_PERMS, &qattr);
	if (!MQ_ID_IS_VALID(memb->mq_id)) {
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}
	event.sigev_notify = SIGEV_THREAD; // async
	event.sigev_notify_function = process_message;
	event.sigev_value.sival_ptr = memb; // member state for this queue
	err = mq_notify(memb->mq_id, &event);
	if (err < 0) {
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}
	// Add the member as the very last operation (else we might have to remove
	// it again in the fail code below).
	group_add_member(group, memb);
	pthread_mutex_unlock(&group->lock);
	*new_memb = memb;
	return 0;

fail:
	// group lock should not be held here
	if (memb) {
		if (MQ_ID_IS_VALID(memb->mq_id)) {
			mq_close(memb->mq_id);
			mq_unlink(memb->mq_name);
		}
		free(memb);
	}
	return exit_errno;
}

static int
group_member_leave(struct group *group, pid_t pid)
{
	int err, exit_errno;
	struct member *member;

	// Lock group, remove group from list, unlock group. Then we can take our
	// time to deallocate the group and its resources.
	pthread_mutex_lock(&group->lock);
	member = group_get_member(group, pid);
	if (!member) {
		pthread_mutex_unlock(&group->lock);
		exit_errno = -ENOENT;
		goto fail;
	}
	__group_rm_member(group, member);
	pthread_mutex_unlock(&group->lock);

	// Indicate we no longer will use the message queue
	err = mq_close(member->mq_id);
	if (err < 0) {
		exit_errno = -(errno);
		goto fail;
	}
	// Destroy the message queue; the kernel will deallocate it once everyone
	// who has opened it has closed it
	err = mq_unlink(member->mq_name);
	if (err < 0) {
		exit_errno = -(errno);
		goto fail;
	}
	free(member);
	return 0;

fail:
	return exit_errno;
}

/*
 * Group list operations.
 */

#define groups_for_each_group(groups, group)	\
	list_for_each_entry(group, &(groups)->list, link)

static inline bool
groups_empty(struct groups *groups)
{
	return list_empty(&groups->list);
}

static inline void
groups_add_group(struct groups *groups, struct group *group)
{
	list_add(&(group->link), &(groups->list));
}

static inline struct group *
groups_get_group(struct groups *groups, const char *key)
{
	struct group *group;
	groups_for_each_group(groups, group) {
		if (strcmp(key, group->user_key) == 0)
			return group;
	}
	return NULL;
}

static inline void
__groups_rm_group(struct groups *groups, struct group *group)
{
	list_del(&(group->link));
}

static inline void
groups_rm_group(struct groups *groups, const char *key)
{
	struct group *group = groups_get_group(groups, key);
	if (group)
		__groups_rm_group(groups, group);
}

/*
 * Misc
 */

static inline const char *
proto_str(proto_code c)
{
	// TODO Fix up these error messages to be more meaningful.
	switch (c) {
		case PROTO_WRONG_WD:
			return "Wrong Watch Descriptor";
		case PROTO_ZERO_READ_LEN:
			return "Zero Length Returned on Read";
		case PROTO_ZERO_EVT_LEN:
			return "Zero Length on Event";
		case PROTO_UNSUPPORTED_EVT_MASK:
			return "Unsupported Event Mask";
		default:
			return "Inotify Protocol Error Unknown";
	}
}

/*-------------------------------------- INTERNAL THREADING ------------------*/

//! Function registered to all message queue notify callbacks.
static void
process_message(union sigval sval)
{
	// this function is registered for all message queue callbacks, thus we
	// extract our specific state from the argument
	//struct group *group = (struct group *)sval.sival_ptr;
	printf("%s\n", __func__);
}

static void
inotify_thread_cleanup(void *arg)
{
	struct inotify_thread_state *state = 
		(struct inotify_thread_state *)arg;

	state->is_alive = false;

	// Check the reason for our exit
	if (state->exit_code != 0) {
		if (state->exit_code != -EPROTO)
			fprintf(stderr, "Membership watch exited fatally: %s\n",
					strerror(-(state->exit_code)));
		else
			fprintf(stderr, "Membership watch exited unexpectedly: %s\n",
					proto_str(state->proto_error));
	}

	if (state->inotify_fd > -1)
		close(state->inotify_fd); // closes watch descriptors, too
	state->inotify_fd = -1;

	if (!state->events)
		free(state->events);
	state->events = NULL;
}

static void *
inotify_thread(void *arg)
{
	struct inotify_event *event; //! Current event we're processing
	struct inotify_thread_state *state =
		(struct inotify_thread_state *)arg;
	bool exit_loop = false; //! Control variable for exiting the main while
	int wd; //! inotify watch descriptor
	int len, err;
	struct group *group = state->group;

	pthread_cleanup_push(inotify_thread_cleanup, state);
	state->is_alive = true;

	/*
	 * Source: www.linuxjournal.com/article/8478
	 */
	state->events = calloc(1, INOTIFY_BUFFER_SZ);
	if (!state->events) {
		state->exit_code = -ENOMEM;
		pthread_exit(NULL);
	}
	state->inotify_fd = inotify_init();
	if (state->inotify_fd < 0) {
		state->exit_code = -(errno);
		pthread_exit(NULL);
	}
	wd = inotify_add_watch(state->inotify_fd,
			state->group->memb_dir, IN_CREATE | IN_DELETE);
	if (wd < 0) {
		state->exit_code = -(errno);
		pthread_exit(NULL);
	}
	while (exit_loop == false) {
		// sleep until next event
		len = read(state->inotify_fd, state->events, INOTIFY_BUFFER_SZ);
		if (len < 0) {
			if (errno == EINTR)
				continue; // read() interrupted by signal, just re-read()
			state->exit_code = -(errno);
			exit_loop = true;
			break;
		} else if (len == 0) {
			state->exit_code = -EPROTO;
			state->proto_error = PROTO_ZERO_READ_LEN;
			exit_loop = true;
			break;
		}
		// length is okay, loop through all events
		int i = 0;
		while (i < len && (exit_loop == false)) {
			event = (struct inotify_event *)(&state->events[i]);
			if (event->len == 0) {
				state->exit_code = -EPROTO;
				state->proto_error = PROTO_ZERO_EVT_LEN;
				exit_loop = true;
				break;
			}
			if (event->wd != wd) {
				state->exit_code = -EPROTO;
				state->proto_error = PROTO_WRONG_WD;
				exit_loop = true;
				break;
			}
			switch (event->mask) {
				case IN_CREATE:
				{
					pid_t pid = atoi(event->name);
					struct member *new_memb;
					err = group_member_join(state->group, pid, &new_memb);
					if (err < 0) {
						// stop on fault; either the member code is broken or
						// someone is manually screwing with our directories
						// and/or files. member code should prevent a client
						// from successfully calling shmgrp_join more than once
						// on the same group.
						state->exit_code = err;
						exit_loop = true;
						break;
					}
					group->notify(MEMBERSHIP_JOIN, new_memb->pid);
				}
				break;
				case IN_DELETE:
				{
					pid_t pid = atoi(event->name);
					err = group_member_leave(group, pid);
					if (err < 0) {
						// stop on fault; either the member code is broken or
						// someone is manually screwing with our directories
						// and/or files. member code should prevent a client
						// from successfully calling shmgrp_leave more than once
						// on the same group.
						state->exit_code = err;
						exit_loop = true;
					}
					group->notify(MEMBERSHIP_LEAVE, pid);
				}
				break;
				default:
				{
					state->exit_code = -EPROTO;
					state->proto_error = PROTO_UNSUPPORTED_EVT_MASK;
					exit_loop = true;
				}
				break;
			}
			i += INOTIFY_EVENT_SZ + event->len;
		}
		// Check the exit flag here again, just in case more code is added below
		// this point. In other words, don't add code between the end of the
		// above while-loop and this if-statement.
		if (exit_loop == true)
			break;
	}

	pthread_cleanup_pop(1); // invoke cleanup routine
	pthread_exit(NULL); // technically not reachable
}

/*-------------------------------------- PUBLIC INTERFACE --------------------*/

int shmgrp_init_leader(void)
{
	int err, exit_errno;
	if (groups) {
		exit_errno = -EEXIST;
	}
	groups = calloc(1, sizeof(*groups));
	if (!groups) {
		exit_errno = -ENOMEM;
	}
	INIT_LIST_HEAD(&groups->list);
	pthread_mutex_init(&groups->lock, NULL);
	// create the shmgrp root membership directory
	err = mkdir(MEMB_DIR_PREFIX_DIR, MEMB_DIR_PERMS);
	if (err < 0 && errno != EEXIST) { // okay if already exists
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

int shmgrp_tini_leader(void)
{
	// FIXME clear other resources within each existing group
	if (groups)
		free(groups);
	groups = NULL;
	// Don't bother deleting the base member directory.
	return 0;
}

int shmgrp_open(const char *key, membership_callback func)
{
	int err, exit_errno;
	struct group *new_grp = NULL;
	struct stat statbuf;
	struct inotify_thread_state *tstate = NULL;

	if (!verify_userkey(key) || !func) {
		exit_errno = -EINVAL;
		goto fail;
	}

	memset(&statbuf, 0, sizeof(struct stat));

	// To prevent multiple threads from opening a group with the same group key
	// (FIXME prevent other processes from opening the same group) we lock the
	// group list and attempt to locate the key. If found, exit. Else continue
	// setting up the group with the lock held. If we instead add the group,
	// unlock the list then proceed to finish initializing the group it will be
	// in an inconsistent state for some time, which we do not want.
	pthread_mutex_lock(&(groups->lock));
	new_grp = groups_get_group(groups, key);
	if (new_grp) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -EEXIST;
		goto fail;
	}
	new_grp = calloc(1, sizeof(*new_grp));
	if (!new_grp) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -ENOMEM;
		goto fail;
	}
	strncpy(new_grp->user_key, key, strlen(key));

	// Construct the membership directory path and verify that it does not
	// already exist.
	snprintf(new_grp->memb_dir, MAX_LEN, MEMB_DIR_FMT, key);
	err = stat(new_grp->memb_dir, &statbuf); // only allow errno of ENOENT
	if (err < 0 && errno != ENOENT) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -(errno); // possible internal error, maybe don't expose this?
		goto fail;
	} else if (err >= 0) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -EEXIST;
		goto fail;
	}
	err = mkdir(new_grp->memb_dir, MEMB_DIR_PERMS);
	if (err < 0) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -(errno); // possible internal error, maybe don't expose this?
		goto fail;
	}

	// Begin accepting registration requests. Spawn an inotify thread on the
	// membership directory we just created.
	tstate = calloc(1, sizeof(*tstate));
	if (!tstate) {
		pthread_mutex_unlock(&(groups->lock));
		exit_errno = -ENOMEM;
		goto fail;
	}
	// cross link thread and group
	tstate->group = new_grp;
	new_grp->inotify_thread = tstate;
	// init remaining useful group fields
	INIT_LIST_HEAD(&new_grp->link);
	INIT_LIST_HEAD(&new_grp->member_list);
	new_grp->notify = func;
	err = pthread_create(&(tstate->tid), NULL, inotify_thread, tstate);
	if (err < 0) {
		exit_errno = -(errno); // possible internal error, maybe don't expose this?
		goto fail;
	}

	groups_add_group(groups, new_grp);

	pthread_mutex_unlock(&groups->lock);

	return 0;

fail:
	if (tstate)
		free(tstate);
	if (new_grp)
		free(new_grp);
	// No need to cancel the inotify thread, because we can never get to the
	// label fail if the thread was successfully created :)
	return exit_errno;
}

int shmgrp_close(const char *key)
{
	int err, exit_errno = 0;
	struct group *group;
	char path[MAX_LEN];

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	memset(path, 0, MAX_LEN);

	// Look up key. Remove group from list if found to ensure other threads
	// closing the same group key do not find it. No need to protect against
	// other processes closing this group, because they would not have had
	// opened the group.
	pthread_mutex_lock(&groups->lock);
	groups_for_each_group(groups, group) {
		if (strcmp(key, group->user_key) == 0)
			break;
	}
	if (!group || strcmp(key, group->user_key) != 0) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	__groups_rm_group(groups, group);
	pthread_mutex_unlock(&groups->lock);

	// Now we are free to operate on the group without further locking.

	// Prevent further registrations
	group_stop_inotify(group);

	// If there are still members in the group, clean up their state
	if (group_has_members(group)) { // What TODO?
		// go through all members and terminate all our state relating to these
		// members, such as mmaped regions, etc
		// delete all the inotify files?? that might be good
	}

	// Grab the path before releasing memory resources.
	snprintf(path, MAX_LEN, MEMB_DIR_FMT, group->user_key);
	free(group->inotify_thread);
	free(group);

	// Attempt to remove the membership directory. We assume at this point that
	// it is empty.
	err = rmdir(path);
	if (err < 0) {
		if (errno == ENOTEMPTY)
			fprintf(stderr, "Could not delete member directory '%s',"
					" delete manually\n", path);
		exit_errno = -(errno);
		goto fail;
	}

	return 0;

fail:
	return exit_errno;
}

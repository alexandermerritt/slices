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
#include <sys/mman.h>
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
	char shm_file[MAX_LEN];
	shmgrp_region_id id; // leader creates the ID
};

// forward declaration
struct group;

struct member
{
	struct list_head link;
	struct list_head region_list;
	pthread_mutex_t lock; //! Region lock
	pid_t pid;
	mqd_t mq_in;
	mqd_t mq_out;
	char mqname_in[MAX_LEN];
	char mqname_out[MAX_LEN];
	shmgrp_region_id next_region_id;
	shm_callback notify;
	pthread_mutex_t mq_lock; //! Lock held when processing messages
	const char * user_key; // points to char user_key in group
};

// forward declaration
struct inotify_thread_state;

struct group
{
	struct list_head link;
	char user_key[SHMGRP_MAX_KEY_LEN];
	char memb_dir[MAX_LEN];
	struct list_head member_list;
	pthread_mutex_t lock; //! Member lock
	group_callback notify;
	struct inotify_thread_state *inotify_thread;

	// State to handle forking.
	bool willfork; //! Value of willfork in shmgrp_open.
	pid_t pid; //! PID of process which created this group.
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

static inline shmgrp_region_id
member_next_region_id(struct member *memb)
{
	// obviously not thread-safe
	return (memb->next_region_id++);
}

static inline void
member_add_region(struct member *mem, struct region *reg)
{
	list_add(&(reg->link), &(mem->region_list));
}

static inline struct region *
member_get_region(struct member *memb, shmgrp_region_id id)
{
	struct region *region;
	member_for_each_region(memb, region)
		if (region->id == id)
			return region;
	return NULL;
}

static inline void
__member_rm_region(struct region *region)
{
	list_del(&region->link);
}

static inline void
member_rm_region(struct member *memb, shmgrp_region_id id)
{
	struct region *region;
	region = member_get_region(memb, id);
	if (region)
		__member_rm_region(region);
}

static int
member_create_region(struct member *memb, size_t size, struct region **region)
{
	int err, exit_errno;
	struct region *reg;
	reg = calloc(1, sizeof(*reg));
	if (!reg) {
		exit_errno = -ENOMEM;
		goto fail;
	}

	pthread_mutex_lock(&memb->lock);

	reg->id = member_next_region_id(memb); // FIXME Need to lock this??
	INIT_LIST_HEAD(&reg->link);
	reg->size = size;

	// Map in the shm file
	snprintf(reg->shm_file, MAX_LEN, SHM_NAME_FMT,
			memb->user_key, memb->pid, reg->id);
	reg->fd = shm_open(reg->shm_file, SHM_OPEN_LEADER_FLAGS, SHM_PERMS);
	if (reg->fd < 0){
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -(errno);
		goto fail;
	}
	reg->addr = mmap(NULL, reg->size,
			MMAP_PERMS, MMAP_FLAGS, reg->fd, 0);
	if (reg->addr == MAP_FAILED) {
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -(errno);
		goto fail;
	}
	// Enlarge region from zero to the correct size
	err = ftruncate(reg->fd, reg->size);
	if (err < 0) {
		exit_errno = -(errno);
		goto fail;
	}

	member_add_region(memb, reg);
	*region = reg;
	pthread_mutex_unlock(&memb->lock);
	return 0;

fail:
	if (reg) {
		if (reg->addr != MAP_FAILED)
			munmap(reg->addr, reg->size);
		if (reg->fd >= 0)
			close(reg->fd);
		free(reg);
	}
	return exit_errno;
}

static inline bool
member_has_regions(struct member *mem)
{
	return !list_empty(&(mem->region_list));
}

// If it cannot find the region ID, error is returned.
static int
member_remove_region(struct member *memb, shmgrp_region_id id)
{
	int err, exit_errno;
	struct region *region;

	pthread_mutex_lock(&memb->lock);

	region = member_get_region(memb, id);
	if (!region) {
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	err = munmap(region->addr, region->size);
	if (err < 0) {
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -(errno);
		goto fail;
	}
	err = close(region->fd);
	if (err < 0) {
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -(errno);
		goto fail;
	}
	err = shm_unlink(region->shm_file);
	if (err < 0) {
		pthread_mutex_unlock(&memb->lock);
		exit_errno = -(errno);
		goto fail;
	}
	__member_rm_region(region);
	free(region);
	pthread_mutex_unlock(&memb->lock);
	return 0;

fail:
	return exit_errno;
}

// Send a message. If we receive some signal while sending, we retry the send.
// If the out MQ is full, we spin trying to send until the MQ accepts it.
static int
member_send_message(struct member *memb, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_send(memb->mq_out, (char*)msg, sizeof(*msg), MQ_DFT_PRIO);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
		if (errno == EAGAIN)
			goto again; // MQ is full, try again (spin)
		exit_errno = -(errno);
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

// Receive a message. If we are sent some signal while receiving, we retry the
// receive. If the in MQ is empty, we return with -EAGAIN instead of spinning.
static int
member_recv_message(struct member *memb, struct message *msg)
{
	int err, exit_errno;
again:
	err = mq_receive(memb->mq_in, (char*)msg, sizeof(*msg), NULL);
	if (err < 0) {
		if (errno == EINTR)
			goto again; // A signal interrupted the call, try again
		exit_errno = -(errno); // must do this, as EAGAIN indicates empty queue
		goto fail;
	}
	return 0;
fail:
	return exit_errno;
}

// forward declaration
static void process_messages(union sigval);

static inline int
member_set_notify(struct member *memb)
{
	int err;
	struct sigevent event;
	memset(&event, 0, sizeof(event));
	event.sigev_notify = SIGEV_THREAD;
	event.sigev_notify_function = process_messages;
	event.sigev_value.sival_ptr = memb;
	err = mq_notify(memb->mq_in, &event);
	if (err < 0)
		return -(errno);
	return 0;
}

/* Call this when we expect there may be messages on the queue. If there are
 * none, this function does nothing. This function must be thread-safe. */
void member_process_messages(struct member *memb)
{
	int err;
	struct message msg;
	struct region *region = NULL;

	pthread_mutex_lock(&memb->mq_lock);

	// Re-enable notification on the message queue. The man page says
	// notifications are one-shot, and need to be reset after each trigger. It
	// additionally says that notifications are only triggered upon receipt of a
	// message on an empty queue. Thus we set notification first, then empty the
	// queue completely.
	err = member_set_notify(memb);
	if (err < 0) {
		fprintf(stderr, "Error setting notify on member %d: %s\n",
				memb->pid, strerror(-(err)));
		pthread_mutex_unlock(&memb->mq_lock);

		return;
	}

	// Pull all messages out until the queue is empty. For each message, process
	// it (either add or remove a region) and invoke the group callback with the
	// new state.
	while (1) {
		err = member_recv_message(memb, &msg); // does not block if MQ is empty
		if (err < 0) {
			if (err == -EAGAIN)
				break; // MQ is empty, notify will trigger next time
			fprintf(stderr, "Error recv msg from memb %d: %s\n",
					memb->pid, strerror(-(err)));
		}

		// we shouldn't have to lock creating/destroying regions as there is
		// only one notification thread per member which calls this function

		if (msg.type == MESSAGE_CREATE_SHM) {
			err = member_create_region(memb, msg.m.region.size, &region);
			if (err < 0) {
				msg.m.region.status = err;
			} else {
				msg.m.region.id = region->id;
				msg.m.region.status = 0; // = okay
			}
			err = member_send_message(memb, &msg); // spins on send unless error
			if (err < 0)
				fprintf(stderr, "Error sending create shm reply to %d: %s\n",
						memb->pid, strerror(-(err)));
			// notify last
			memb->notify(SHM_CREATE_REGION, memb->pid, msg.m.region.id);
		}

		else if (msg.type == MESSAGE_REMOVE_SHM) {
			// If willfork=true, the child process needs to make sure that it
			// does not introduce a race condition between us (this function on
			// the MQ notify thread stack) and another thread in the child,
			// which might presumably call destroy_member; destroy_member will
			// lock the member as it removes it, preventing us from removing the
			// region (which requires holding the same member lock) and
			// unmapping resources. The root leader should instead signal all
			// children to quit after it receives a MEMBERSHIP_LEAVE, assuming
			// the application has rmreg'd all regions.
			memb->notify(SHM_REMOVE_REGION, memb->pid, msg.m.region.id);
			msg.m.region.status = member_remove_region(memb, msg.m.region.id);
			err = member_send_message(memb, &msg); // spins on send unless error
			if (err < 0)
				fprintf(stderr, "Error sending remove shm reply to %d: %s\n",
						memb->pid, strerror(-(err)));
			break;
		}
	}
	pthread_mutex_unlock(&memb->mq_lock);
}

/*
 * Group operations.
 */

#define group_for_each_member(group, member)	\
	list_for_each_entry(member, &((group)->member_list), link)

static inline void
__group_add_member(struct group *group, struct member *memb)
{
	list_add(&(memb->link), &(group->member_list));
	memb->user_key = group->user_key;
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
__group_rm_member(struct member *member)
{
	list_del(&member->link);
	member->user_key = NULL;
}

static inline int
group_rm_member(struct group *group, pid_t pid)
{
	struct member *member;
	member = group_get_member(group, pid);
	if (member)
		__group_rm_member(member);
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

// Locks the group. Verifies member pid doesn't already exist, creates new
// member, opens all MQs and adds member to group.
//
// This function performs a sequence of steps whose ordering is extremely vital,
// because it must perform actions LOCK-STEP with what happens or what MAY
// HAPPEN in the member (join, then mkreg or leave in any order, any number of
// times).
static int
group_member_join(struct group *group, pid_t pid,
		shm_callback func, struct member **new_memb)
{
	int exit_errno;
	struct mq_attr qattr;
	struct member *memb = NULL;

	memset(&qattr, 0, sizeof(qattr));

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
	memb->mq_in = MQ_ID_INVALID_VALUE;
	memb->mq_out = MQ_ID_INVALID_VALUE;
	memb->notify = func;
	memb->next_region_id = 1;
	pthread_mutex_init(&memb->mq_lock, NULL);
	pthread_mutex_init(&memb->lock, NULL);

	// ----1----
	// Add the member to the group. This MUST be done prior to both setting
	// notification on MQ-in and allowing the member to wake up (creating
	// MQ-out). The reason is because once notify is enabled and the member
	// woken up, the member may send a message to us. This will trigger the
	// notify thread to process any messages from the member, which involves
	// invoking the member callback routine. Within this routine, the application
	// may request we LOOKUP the region we just created. However if the member
	// has not yet been added to the group, that will fail. So we add the member
	// to the group here.
	__group_add_member(group, memb);

	// ----2----
	// OPEN the MQ under ownership of the member (MQ-in). If this fails, then
	// the PID reported to us is not actually requesting to join the group
	// (because in join() it should have created our MQ-in, or from its
	// perspective, MQ-out). Thus if mq_open fails with errno ENOENT, it merely
	// means that the group_callback within the application has borked its value
	// of pid, somehow. But that is unlikely.
	//
	// The member process is still blocked in its call to join() at this point,
	// waiting for us to CREATE MQ-out (what it calls MQ-in within member.c).
	snprintf(memb->mqname_in, MAX_LEN, MQ_NAME_FMT_MEMBER, group->user_key, pid);
	memb->mq_in = mq_open(memb->mqname_in, MQ_OPEN_CONNECT_FLAGS);
	if (!MQ_ID_IS_VALID(memb->mq_in)) {
		__group_rm_member(memb);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// ----3----
	// Enable notification on our MQ-in BEFORE we create our MQ-out.
	// Notification is ONLY triggered when a message arrives to an EMPTY MQ.
	// Thus we enable notification before the member wakes up to prevent it from
	// injecting a message on our MQ-in before we're able to set notification.
	// The member wakes up once we create our MQ-out (below).
	member_set_notify(memb);

	// ----4----
	// Create the MQ under our ownership. Owned MQs are used for sending. This
	// wakes up the member, allowing it to return from a call to join(). Thus
	// ANYTIME AFTER THIS POINT the member application may send messages on
	// MQ-in or request to leave the group.
	qattr.mq_maxmsg = MQ_MAX_MESSAGES;
	qattr.mq_msgsize = MQ_MAX_MSG_SIZE;
	snprintf(memb->mqname_out, MAX_LEN, MQ_NAME_FMT_LEADER,
			memb->user_key, memb->pid);
	memb->mq_out = mq_open(memb->mqname_out, MQ_OPEN_OWNER_FLAGS, MQ_PERMS, &qattr);
	if (!MQ_ID_IS_VALID(memb->mq_out)) {
		__group_rm_member(memb);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}

	// If the member application decided to withdraw its membership from us
	// RIGHT NOW, it will delete (unlink) its membership file from the
	// directory. Since this function will only exist on the inotify thread
	// stack, and the inotify thread stack (of which there exists only ONE) is
	// the only one to call either group_member_join and group_member_leave,
	// there is no worry about both functions being on separate simultaneously
	// executing stacks at the same time for the same member.

	pthread_mutex_unlock(&group->lock);
	if (new_memb)
		*new_memb = memb;
	return 0;

fail:
	// group lock should not be held here
	if (memb) {
		if (MQ_ID_IS_VALID(memb->mq_in)) {
			mq_close(memb->mq_in);
		}
		if (MQ_ID_IS_VALID(memb->mq_out)) {
			mq_close(memb->mq_out);
			mq_unlink(memb->mqname_out);
		}
		free(memb);
	}
	return exit_errno;
}

// Locks the group, removes a member, destroys member and all resources.
static int
group_member_leave(struct group *group, pid_t pid)
{
	int err, exit_errno;
	struct member *member;

	// Lock member list, remove member from list, deallocate member resources,
	// deallocate member, unlock member list.
	pthread_mutex_lock(&group->lock);
	member = group_get_member(group, pid);
	if (!member) {
		pthread_mutex_unlock(&group->lock);
		exit_errno = -ENOENT;
		goto fail;
	}

	// We must also lock the member, as an MQ notify thread may additionally be
	// removing regions from this member.
	pthread_mutex_lock(&member->lock);

	if (group->willfork && (group->pid == getpid())) {
		// Client has multi-process leader, and is calling destroy_member from
		// the root leader instead of the child which established the member.
		// establish_member creates MQs, which if done in the child process, do
		// not exist in the parent leader.
		pthread_mutex_unlock(&member->lock);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -EPROTO;
		goto fail;
	}

	if (member_has_regions(member)) {
		// FIXME What to do? These should all be cleared. One way these may
		// still exist is if the member process does not remove its regions with
		// us.
		fprintf(stderr, "Warning: memory regions still"
				" exist with member %d\n", member->pid);
	}

	__group_rm_member(member);

	// Indicate we no longer will use the message queues
	err = mq_close(member->mq_in);
	if (err < 0) {
		pthread_mutex_unlock(&member->lock);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}
	err = mq_close(member->mq_out);
	if (err < 0) {
		pthread_mutex_unlock(&member->lock);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}
	// Destroy our message queue; the kernel will deallocate it once everyone
	// who has opened it has closed it
	err = mq_unlink(member->mqname_out);
	if (err < 0) {
		pthread_mutex_unlock(&member->lock);
		pthread_mutex_unlock(&group->lock);
		exit_errno = -(errno);
		goto fail;
	}

	free(member);
	pthread_mutex_unlock(&group->lock);
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
	groups_for_each_group(groups, group)
		if (strcmp(key, group->user_key) == 0)
			return group;
	return NULL;
}

static inline void
__groups_rm_group(struct group *group)
{
	list_del(&(group->link));
}

static inline void
groups_rm_group(struct groups *groups, const char *key)
{
	struct group *group = groups_get_group(groups, key);
	if (group)
		__groups_rm_group(group);
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

/**
 * Function registered to ALL message queue notify callbacks. It must be
 * thread-safe.
 */
static void
process_messages(union sigval sval)
{
	struct member *memb;
	// Extract our specific state from the argument; no locking needed.
	memb = (struct member*) sval.sival_ptr;
	member_process_messages(memb);
}

/**
 * Arguments to group_callback + pointer to notify function. Used by the thread
 * spawned to invoke that callback.
 */
struct group_callback_args
{
	group_event e;
	pid_t pid;
	struct group *group;
};

/**
 * Thread created for each invocation of the group_callback when the inotify
 * thread created for the group observes an event.
 *
 * The reason we spawn a thread, is in case the user does a fork() inside
 * group_notify. fork() recreates only the calling thread in the process. Should
 * we not create a thread to invoke the callback, the inotify thread will exist
 * in the child and will return: there will then be an additional thread
 * observing for group joins/leaves.  This may not be expected behavior. We can
 * always not create a thread in the future and do something else.
 *
 * TODO In the future, we can selectively spawn a thread to handle the notify
 * callback if we see group.willfork is set to true to reduce overhead.
 */
static void *
group_notify_thread(void *arg)
{
	struct group_callback_args *args = (struct group_callback_args*)arg;
	args->group->notify(args->e, args->pid);
	free(args);
	return NULL;
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
	pid_t pid;

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

			// Spawn a new thread to invoke the group_callback given to us for
			// the group. Give the thread its own argument structure which it
			// will deallocate.
			pid = atoi(event->name);
			pthread_t dontneed;
			struct group_callback_args *args = calloc(1, sizeof(*args));
			if (!args) {
				state->exit_code = -ENOMEM;
				exit_loop = true;
				break;
			}
			args->group = group;
			args->pid = pid;
			switch (event->mask) {

				case IN_CREATE:
					args->e = MEMBERSHIP_JOIN;
					// group_member_join is invoked via a function exposed to
					// the caller; they decide whether or not to accept it.
					break;

				case IN_DELETE:
					args->e = MEMBERSHIP_LEAVE;
					break;

				default:
					state->exit_code = -EPROTO;
					state->proto_error = PROTO_UNSUPPORTED_EVT_MASK;
					exit_loop = true;
					break;
			}
			if (exit_loop == true)
				break;
			err = pthread_create(&dontneed, NULL, group_notify_thread, (void*)args);
			if (err < 0) {
				state->exit_code = -ENOMEM;
				exit_loop = true;
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
		return -EEXIST; // don't go to fail, as groups will be deallocated
	}
	groups = calloc(1, sizeof(*groups));
	if (!groups) {
		exit_errno = -ENOMEM;
		goto fail;
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
	if (groups)
		free(groups);
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

int shmgrp_open(const char *key, group_callback func, bool willfork)
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
	pthread_mutex_init(&new_grp->lock, NULL);
	new_grp->notify = func;
	new_grp->willfork = willfork;

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

	// Look up key. Remove group from list.
	pthread_mutex_lock(&groups->lock);
	group = groups_get_group(groups, key);
	if (!group) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	__groups_rm_group(group);

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
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -(errno);
		goto fail;
	}

	pthread_mutex_unlock(&groups->lock);
	return 0;

fail:
	return exit_errno;
}

int shmgrp_establish_member(const char *key, pid_t pid, shm_callback func)
{
	int err, exit_errno = 0;
	struct group *group = NULL;

	if (!verify_userkey(key) || !func) {
		exit_errno = -EINVAL;
		goto fail;
	}

	pthread_mutex_lock(&groups->lock);

	group = groups_get_group(groups, key);
	if (!group) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// Verify consistency of willfork.
	if (group->willfork && (group->pid == getpid())) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EPROTO;
		goto fail;
	}

	err = group_member_join(group, pid, func, NULL);
	if (err < 0) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = err;
		goto fail;
	}

	pthread_mutex_unlock(&groups->lock);
	return 0;

fail:
	return exit_errno;
}

int shmgrp_destroy_member(const char *key, pid_t pid)
{
	int err, exit_errno = 0;
	struct group *group = NULL;

	if (!verify_userkey(key)) {
		exit_errno = -EINVAL;
		goto fail;
	}

	pthread_mutex_lock(&groups->lock);

	group = groups_get_group(groups, key);
	if (!group) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}
	err = group_member_leave(group, pid);
	if (err < 0) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = err;
		goto fail;
	}

	pthread_mutex_unlock(&groups->lock);
	return 0;

fail:
	return exit_errno;
}

int shmgrp_member_region(const char *key, pid_t pid,
		shmgrp_region_id id, struct shmgrp_region *reg)
{
	int exit_errno = 0;
	struct group *group;
	struct member *member;
	struct region *region;

	// Unfortunately this function can get complicated if we wish to play it
	// safe. We must be able to perform a lookup without crashing even if
	// members are leaving or closing their regions.
	//
	// Thankfully, pairs of functions control the addition and removal of
	// elements to and from objects, locking those objects while doing so. i.e.
	// adding a group means locking the group list; adding a member to a group
	// means locking the group; etc.  So, hopefully we won't crash if we just
	// lock everything top-down. In the event of failures, unlock what was
	// locked in reverse order.
	//
	// Note that it may very well be possible that the client code invokes this
	// function from within their shm_callback! This routine is called from
	// process_messages which only locks mq_lock in a member, which we do not
	// use.

	if (!verify_userkey(key) || !reg) {
		exit_errno = -EINVAL;
		goto fail;
	}

	// Lock group list, get group
	pthread_mutex_lock(&groups->lock);
	group = groups_get_group(groups, key);
	if (!group) {
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// Lock group, get member
	pthread_mutex_lock(&group->lock);
	member = group_get_member(group, pid);
	if (!member) {
		pthread_mutex_unlock(&group->lock);
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// Lock member, get region
	pthread_mutex_lock(&member->lock);
	region = member_get_region(member, id);
	if (!region) {
		pthread_mutex_unlock(&member->lock);
		pthread_mutex_unlock(&group->lock);
		pthread_mutex_unlock(&groups->lock);
		exit_errno = -EINVAL;
		goto fail;
	}

	// Copy region state to output parameter
	reg->id = region->id;
	reg->addr = region->addr;
	reg->size = region->size;

	pthread_mutex_unlock(&member->lock);
	pthread_mutex_unlock(&group->lock);
	pthread_mutex_unlock(&groups->lock);
	return 0;

fail:
	return exit_errno;
}

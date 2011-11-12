/**
 * @file shmgrp.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-06
 * @brief TODO
 *
 * ALL functions return 0 (zero) on success and less than 0 if not successful.
 * Values less than zero are actually negative errno codes to indicate what went
 * wrong. But they're not clean right now, as I expose low-level errno codes
 * directly returned from system calls within the library as the return code of
 * an API routine. However, you can be pretty sure that if you get
 * 		-EINVAL, -EEXIST, or -ENOENT
 * it means you are not passing valid arguments.
 *
 * TODO LIST ASSUMPTIONS, LIMITATIONS, RESTRICTIONS HERE.
 */

#ifndef _SHMGRP_H
#define _SHMGRP_H

#include <stdbool.h>
#include <unistd.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

/**
 * Maximum number of characters allowable in a group key. Group keys should
 * contain printable ASCII characters minus the following: / and space
 */
#define SHMGRP_MAX_KEY_LEN	128

/**
 * ID type for a shared memory region. Region IDs are unique within each
 * group-member tuple.
 */
typedef int shmgrp_region_id;

/**
 * Structure describing a mapped region. Do not munmap the virtual address or I
 * will cut your face off.
 */
struct shmgrp_region
{
	shmgrp_region_id id;
	void *addr;
	size_t size;
};

/**
 * Event types group_callback will be given when invoked by the library.
 */
typedef enum { MEMBERSHIP_JOIN = 1, MEMBERSHIP_LEAVE } group_event;

/**
 * Event types shm_callback will be given when invoked by the library.
 */
typedef enum { SHM_CREATE_REGION = 1, SHM_REMOVE_REGION } shm_event;

/**
 * Callback routine which must be defined by the application, and associated
 * with a specific group key upon creation of said group.
 *
 * Each group created should have a separate function defined for the callback.
 * The routine will be invoked when other processes join or leave the group. The
 * PID may be used to differentiate among members in the group.
 *
 * This function must be thread-safe.
 */
typedef void (*group_callback)(group_event, pid_t);

/**
 * Callback routine which must be defined by the application, and associated
 * with a specific member of a group when that member has been granted access to
 * join the group.
 *
 * One function may be used for all members of a group, as the PID is a provided
 * parameter. The region ID parameter represents a region that has already been
 * mapped in and its associated virtual memory region written to or read from.
 * This function is invoked when a member process calls shmgrp_mkreg.
 *
 * This function must be thread-safe.
 */
typedef void (*shm_callback)(shm_event, pid_t, shmgrp_region_id);

/*-------------------------------------- COMMON FUNCTIONS --------------------*/

/**
 * Initialize the shmgrp library state.
 *
 * This must happen once and before any of the below functions are called.
 */
int shmgrp_init(void);

/**
 * Deallocate the shmgrp library state.
 *
 * This must happen last.
 */
int shmgrp_tini(void);

/**
 * Convert the given group_event to a human-readable string for printing.
 */
const char * shmgrp_memb_str(group_event e);

/*-------------------------------------- LEADER FUNCTIONS --------------------*/

/**
 * Open a new group for members to join.
 *
 * The last parameter is important, to help the library deal with forking
 * correctly, as shmgrp is a stateful library. If set to true, it tells the
 * library that you organize your leader as a multi-process runtime, wherein you
 * have a main parent leader that deals with group joins/departures, forking
 * within group_callback to spawn children for each new member. Within the
 * child's stack of group_callback you then must invoke establish_member and
 * destroy_member.  The value passed as willfork applies to the group and all
 * members while the group is open and cannot be changed.  Note that forking is
 * not supported anywhere else in the leader.
 *
 * If willfork is false, then your single-process leader must invoke
 * establish_member and destroy_member itself as it is assumed to additionally
 * be responsible for working directly together with existing members, in
 * addition to handling new/departing members.
 *
 * To put it simply: whichever *process* calls establish_member, that process,
 * and ONLY that process, is responsible for calling destroy_member.
 *
 * @param key	Unique group identifier.
 * @param func	Caller-implemented function to process group joins/departures.
 * @param willfork True if func will fork children to associate with members.
 */
int shmgrp_open(const char *key, group_callback func, bool willfork);

/**
 * Close an existing group and prevent new members from joining.
 *
 * FIXME This will probably break if members still exist when it is called.
 * Ensure they all leave before you shut down.
 */
int shmgrp_close(const char *key);

/**
 * Allow a member to join a group.
 *
 * This function must only be called from within the group_callback routine
 * registered with the creation of a group, and only when the event signifies a
 * new member joining. You must provide a second-level callback routine that
 * will be invoked by accepted members when they create and destroy shared
 * memory regions.
 *
 * @return zero if okay, -EPROTO if you called this from the process which
 * opened the group with willfork=true, -EINVAL/-ENOENT for invalid arguments,
 * else < 0 for internal errors.
 */
int shmgrp_establish_member(const char *key, pid_t pid, shm_callback func);

/**
 * Finalize the departure of a member from a group.
 *
 * This function has the same requirements as establish_member, but must be
 * invoked when the event signifies the departure of an existing member.
 *
 * @return zero if okay, -EPROTO if you called this from a process which did NOT
 * establish the member, -EINVAL/-ENOENT for invalid arguments, else < 0 for
 * internal errors.
 */
int shmgrp_destroy_member(const char *key, pid_t pid);

/**
 * Obtain information about an existing shared memory region established with a
 * member of a group. Valid region IDs come from a parameter passed to
 * invocations of the shm_callback routine.
 *
 * @param	key		Group identifier
 * @param	pid		Member identifier
 * @param	id		Region identifier
 * @param	region	Location where region information will be written.
 */
int shmgrp_member_region(const char *key, pid_t pid,
		shmgrp_region_id id, struct shmgrp_region *region);

/*-------------------------------------- MEMBER FUNCTIONS --------------------*/

/**
 * Request to join a group.
 *
 * This will spin indefinitely if a leader decides not to permit you. A member
 * may not join a group more than once, but it may join and be a member of
 * multiple groups simultaneously.
 */
int shmgrp_join(const char *key);

/**
 * Leave a group.
 *
 * FIXME You should remove all memory regions before doing this. I don't know
 * what will happen if you do not.
 */
int shmgrp_leave(const char *key);

/**
 * Create a shared memory region with your group leader.
 *
 * A member may create shared memory regions only with groups it has been
 * permitted to join. Any number of regions may be created. Each time a region
 * is created, the leader is immediately notified and has that region mapped in
 * its address space. The leader will also know the PID of the member invoking
 * this function. The leader's shm_callback is invoked when a member calls this
 * function.
 *
 * The region ID returned is unique to only the member and the group leader. In
 * other words, it is not unique across groups. However, the value of the ID is
 * common between a member and leader, that is, if a member communicates the
 * region ID out-of-band to the leader (or vice versa) either side can use it to
 * locate the same physical memory mapping.
 *
 * Obviously the key must associate with a group the member has been admitted
 * to.
 */
int shmgrp_mkreg(const char *key, size_t size, shmgrp_region_id *id);

/**
 * Destroy a shared memory mapping.
 *
 * A member and leader should agree to stop using a shared memory region before
 * a member invokes this function. This will fail if the member calling this
 * function has not joined the group represented by the key, or if the ID does
 * not represent any known region.
 */
int shmgrp_rmreg(const char *key, shmgrp_region_id id);

/**
 * Obtain information about an existing shared memory region established with
 * the leader of a group you are a member of. Valid region IDs come from
 * successful calls to shmgrp_mkreg.
 *
 * @param	key		Group identifier
 * @param	id		Region identifier
 * @param	region	Location where region information will be written.
 */
int shmgrp_leader_region(const char *key, shmgrp_region_id id,
		struct shmgrp_region *region);

#endif /* _SHMGRP_H */

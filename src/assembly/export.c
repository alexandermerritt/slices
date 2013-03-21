/**
 * @file export.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-22
 * @brief Implementation separated from assembly.c to support exporting and
 * importing of assembly data across process address spaces. Used to restore
 * state into the assembly module required of of a multi-process runtime. It's
 * separate because a) I don't want to clutter assembly.c with all this, b) it's
 * a workaround for something I hadn't originally intended to implement in the
 * first place.
 */

// System includes
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <uuid/uuid.h>

// Project includes
#include <assembly.h>

// Directory-immediate includes
#include "types.h"

/*-------------------------------------- DEFINITIONS -------------------------*/

//! Minimum len of a char[] needed to hold the string representation of a UUID 
#define UUID_STR_LEN		64

#define EXPORT_KEY_FMT		"assm-export-key-%s"
#define EXPORT_KEY_LEN		255
#define EXPORT_SHM_FLAGS	(O_RDWR | O_CREAT | O_EXCL)
#define EXPORT_SHM_MODE		0660	// -rw-rw----
#define EXPORT_SHM_SIZE		sizeof(struct assembly)
#define EXPORT_MMAP_PERMS	PROT_WRITE
#define EXPORT_MMAP_FLAGS	MAP_SHARED

#define IMPORT_SHM_FLAGS	O_RDWR
#define IMPORT_MMAP_FLAGS	MAP_SHARED
#define IMPORT_MMAP_PERMS	PROT_READ

/*-------------------------------------- EXTERNAL FUNCTIONS ------------------*/

int export_assembly(const uuid_t uuid, struct assembly *assm)
{
	int err;
	char uuid_str[UUID_STR_LEN];
	char shm_file[EXPORT_KEY_LEN];
	void *shm = NULL;
	int shm_fd = -1;

	memset(uuid_str, 0, UUID_STR_LEN);
	memset(shm_file, 0, EXPORT_KEY_LEN);

	uuid_unparse(uuid, uuid_str);
	snprintf(shm_file, EXPORT_KEY_LEN, EXPORT_KEY_FMT, uuid_str);

	// create the shm file and map it in
	shm_fd = shm_open(shm_file, EXPORT_SHM_FLAGS, EXPORT_SHM_MODE);
	if (shm_fd < 0)
		goto fail;
	err = ftruncate(shm_fd, EXPORT_SHM_SIZE); // enlarge the space
	if (err < 0)
		goto fail;
	shm = mmap(NULL, EXPORT_SHM_SIZE, EXPORT_MMAP_PERMS,
			EXPORT_MMAP_FLAGS, shm_fd, 0);
	if (shm == MAP_FAILED)
		goto fail;

	// copy the assembly into it
	memcpy(shm, assm, sizeof(*assm));

	// unmap and close it off but don't delete it
	err = munmap(shm, EXPORT_SHM_SIZE);
	err = close(shm_fd);

	return 0;

fail:
	if (shm != MAP_FAILED)
		munmap(shm, EXPORT_SHM_SIZE);
	if (shm_fd) {
		close(shm_fd);
		shm_unlink(shm_file);
	}
	return -1;
}

int import_assembly(const uuid_t uuid, struct assembly *assm)
{
	int err, exit_errno = -1;
	char uuid_str[UUID_STR_LEN];
	char shm_file[EXPORT_KEY_LEN];
	void *shm = NULL;
	int shm_fd = -1;

	memset(uuid_str, 0, UUID_STR_LEN);
	memset(shm_file, 0, EXPORT_KEY_LEN);

	uuid_unparse(uuid, uuid_str);
	snprintf(shm_file, EXPORT_KEY_LEN, EXPORT_KEY_FMT, uuid_str);

	// open the shm file and map it in
	shm_fd = shm_open(shm_file, IMPORT_SHM_FLAGS, EXPORT_SHM_MODE);
	if (shm_fd < 0) {
		if (errno == ENOENT)
			exit_errno = -ENOENT;
		goto fail;
	}
	err = ftruncate(shm_fd, EXPORT_SHM_SIZE); // enlarge the space
	if (err < 0)
		goto fail;
	shm = mmap(NULL, EXPORT_SHM_SIZE, IMPORT_MMAP_PERMS,
			IMPORT_MMAP_FLAGS, shm_fd, 0);
	if (shm == MAP_FAILED)
		goto fail;

	// copy the assembly out
	memcpy(assm, shm, sizeof(*assm));

	// unmap, close and delete the file
	err = munmap(shm, EXPORT_SHM_SIZE);
	err = close(shm_fd);
	err = shm_unlink(shm_file);

	return 0;
fail:
	if (shm != MAP_FAILED)
		munmap(shm, EXPORT_SHM_SIZE);
	if (shm_fd) {
		close(shm_fd);
		shm_unlink(shm_file);
	}
	return exit_errno;
}

/**
 * @file libregistration.c
 *
 * @date October 2, 2011
 * @author Alex Merritt, merritt.alex@gatech.edu
 *
 * @brief Code herein defines the protocol used between an interposing library
 * and the backend process when initializing state to share memory for passing
 * marshalled CUDA function calls.
 *
 * The backend creates a directory it monitors with inotify.  Files created in
 * REG_DIR by other processes tell the backend a new process wishes to register
 * with the assembly runtime. Deletion of this file by a process tells the
 * backend it is withdrawing. Files have the following format:
 *
 * 	REG_DIR/pid		where 'pid' is the pid of the requesting process
 *
 * Shared memory objects are created by processes requesting participation in
 * the runtime, and are mapped in by the backend. Files are created by the POSIX
 * shared memory interface, shm_[open|unlink], and are placed in /dev/shm/. File
 * names follow the format:
 *
 * 	/dev/shm/REG_OBJ_PREFIXpid	where 'pid' is again like above
 *
 * The backend will destroy the shm file instead of the participating process,
 * as munmap must occur before shm_unlink.
 *
 * Example: process 12345 creates /tmp/assembly/12345 then opens and maps in
 * /dev/shm/asm-12345. The backend is triggered by the former, and maps in the
 * contents of the latter. Process 12345 is done; it munmaps /dev/shm/asm-12345,
 * then deletes /tmp/assembly/12345. This triggers the backend to munmap the shm
 * object, and to finally unlink it.
 */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <glib.h>
#include <pthread.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <debug.h>
#include <common/libregistration.h>

/*-------------------------------------- DEFINITIONS -------------------------*/

/*
 * None of the macros or defined structures in this section are intended to be
 * accessible outside of this file.
 */

#define REG_DIR			"/tmp/assembly"
#define REG_PID			REG_DIR "/.backend.pid"
#define REG_OBJ_PREFIX	"/asm-" //! shm obj filename format; ends with PID

//! Number of events we expect to be queued each time an inotify trigger occurs.
#define INOTIFY_NUM_EVENTS		32
#define INOTIFY_EVENT_SZ		(sizeof(struct inotify_event))
#define INOTIFY_BUFFER_SZ		(INOTIFY_NUM_EVENTS * INOTIFY_EVENT_SZ)

//! Size of the shared memory segment between library and backend.
#define SHM_SZ_MB		8
#define SHM_SZ_B		(SHM_SZ_MB << 20)

#define LOWEST_VALID_REGID	1

/**
 * State representing registration between an interposing library and backend
 * process. The id is _only_ valid within a specific process. Separate
 * registration structures will exist for each individual file/mmap, one in the
 * backend and one in each library (see backend_state and library_state: each
 * have their own array of these).
 */
struct registration {
	regid_t	id;				//! Client-exposed ID
	pid_t	pid;			//! PID of process which created the file
	void	*shm;			//! Address to shared memory area
	int		fd;				//! File descriptor of mmap'd file
	char	shmfile[255];	//! Name of POSIX shm obj
	char	regfile[255];	//! Name of file used to register with backend
};

/**
 * State associated with the inotify thread. This thread performs upcalls (it
 * "calls back") to the backend process via the function the backend provided
 * whenever its inotification state has been triggered.
 */
struct inotify_state {
	//! Upcall to notify backend of registration
	void (*func)(enum callback_event e, regid_t id);
	pthread_t		tid;
	gboolean		alive;
};

/**
 * Registration state maintained by the backend process. No library states
 * exist in the backend.
 */
struct backend_state {
	gboolean				valid; //! true if state has been initialized
	struct registration		**all_regs; //! array of registrations
	unsigned int			max_regs;
	struct inotify_state	inotify;
};

/**
 * Registration state maintained by _each_ process linked with the interposing
 * library.
 */
struct library_state {
	gboolean				valid;
	struct registration		**all_regs;
	unsigned int			max_regs;
};

/*-------------------------------------- GLOBAL STATE ------------------------*/

/*
 * These variables are 'global' with respect to this file only.
 */

static struct backend_state be_state;
static struct library_state lib_state;
static regid_t next_reg_id = LOWEST_VALID_REGID;

/*-------------------------------------- COMMON INTERNAL FUNCTIONS -----------*/

// TODO
// 			static int _reg_mmap(path, *reg);
// 			static int _reg_munmap(path, *reg);

/*-------------------------------------- LIBRARY FUNCTIONS --------------------*/

int reg_lib_init(void) {
	memset(&lib_state, 0, sizeof(struct library_state));
	// @todo TODO don't hard-code size of array
	lib_state.all_regs = calloc(5, sizeof(struct registration *));
	if (!lib_state.all_regs) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	lib_state.max_regs = 5;
	lib_state.valid = TRUE;
	return 0;

fail:
	return -1;
}

/* @todo FIXME this assumes only one file is mmap'd for now, with the pid as the
 * name of the file. if multiple files are to be mmap'd, then we'd have to come
 * up with a different naming scheme for them, like <pid>.<N>
 *
 * @todo FIXME there is no locking here. think of situations where locking would
 * be needed (only when the above fixme is done, of course)
 *
 * @todo TODO Move all the functionality in this code to invoke an internal
 * fucntion. That way, shutdown can invoke the internal function for each reg
 * that still exists.
 */
regid_t reg_lib_connect(void) {
	struct registration *reg = NULL;
	pid_t pid = getpid();
	char filename[256], pid_str[64]; // @todo TODO don't hard-code these
	int fd, err;

	// FIXME only assume one registration at a time
	if (lib_state.all_regs[0]) {
		printd(DBG_ERROR, "Only 1 registration per lib supported\n");
		goto fail;
	}

	reg = calloc(1, sizeof(struct registration));
	if (!reg) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	reg->pid = pid;
	reg->id = next_reg_id++;

	// 1. Create the shm object file.
	memset(filename, 0, sizeof(filename));
	memset(pid_str, 0, sizeof(pid_str));
	sprintf(pid_str, "%d", pid); // convert pid to string
	strcat(filename, REG_OBJ_PREFIX);
	strcat(filename, pid_str);
	strncpy(reg->shmfile, filename, sizeof(filename));
	printd(DBG_DEBUG, "Creating shm obj '%s'\n", reg->shmfile);
	fd = shm_open(reg->shmfile, O_RDWR | O_CREAT | O_EXCL, S_IRWXU | S_IRWXG);
	if (fd < 0) {
		printd(DBG_ERROR, "could not open shm object '%s'\n", reg->shmfile);
		perror("shm_open");
		goto fail;
	}
	err = ftruncate(fd, SHM_SZ_B); // resize it
	if (err < 0) {
		printd(DBG_ERROR, "could not resize shm obj '%s'\n", reg->shmfile);
		perror("ftruncate");
		goto fail;
	}

	// 2. Write a pid file to REG_DIR, triggers inotify in backend.
	memset(filename, 0, sizeof(filename));
	strcat(filename, REG_DIR);
	strcat(filename, "/");
	strcat(filename, pid_str);
	strncpy(reg->regfile, filename, sizeof(filename));
	printd(DBG_DEBUG, "Creating reg file '%s'\n", reg->regfile);
	err = creat(reg->regfile, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP); // rw-rw----
	if (err < 0) {
		printd(DBG_ERROR, "could not creat(%s)\n", reg->regfile);
		perror("creat");
		goto fail;
	}

	// 3. mmap it (backend will simultaneously be mmap'ing)
	reg->fd = fd;
	reg->shm = mmap(NULL, SHM_SZ_B, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (reg->shm == MAP_FAILED) {
		printd(DBG_ERROR, "Could not mmap fd %d\n", fd);
		perror("mmap");
		goto fail;
	}
	lib_state.all_regs[0] = reg;
	printd(DBG_DEBUG, "id=%d pid=%d shm=%p fd=%d\n",
			reg->id, reg->pid, reg->shm, reg->fd);
	return reg->id;

fail:
	if (reg)
		free(reg);
	return -1;
}

void* reg_lib_get_shm(regid_t id) {
	void *shm = NULL;
	if (id < LOWEST_VALID_REGID) {
		printd(DBG_DEBUG, "invalid id %d\n", id);
		return NULL;
	}
	// TODO search for id
	if (lib_state.valid && lib_state.all_regs[0])
		shm = lib_state.all_regs[0]->shm;
	return shm;
}

size_t reg_lib_get_shm_size(regid_t id) {
	size_t size = -1;
	if (id < LOWEST_VALID_REGID) {
		printd(DBG_DEBUG, "invalid id %d\n", id);
		return 0;
	}
	// TODO search for id
	if (lib_state.valid && lib_state.all_regs[0])
		size = SHM_SZ_B;
	return size;
}

int reg_lib_disconnect(regid_t id) {
	struct registration *reg = NULL;
	int err;
	char shmfile[256], pid_str[64];
	memset(shmfile, 0, sizeof(shmfile));
	memset(pid_str, 0, sizeof(pid_str));

	if (id < LOWEST_VALID_REGID) {
		printd(DBG_DEBUG, "invalid id %d\n", id);
		goto fail;
	}
	if (lib_state.valid == FALSE) {
		printd(DBG_WARNING, "lib state not valid\n");
		goto fail;
	}
	if (!lib_state.all_regs) {
		printd(DBG_ERROR, "inconsistent state; valid && !all_regs\n");
		goto fail;
	}
	// TODO search for id
	// TODO move this to a common function
	reg = lib_state.all_regs[0];
	lib_state.all_regs[0] = NULL;
	err = munmap(reg->shm, SHM_SZ_B);
	if (err < 0) {
		printd(DBG_ERROR, "munmap of shm failed: %s\n", strerror(errno));
		goto fail;
	}
	err = shm_unlink(reg->shmfile);
	if (err < 0) {
		printd(DBG_ERROR, "close of shm fd failed: %s\n", strerror(errno));
		goto fail;
	}
	// remove the registration file so the backend deletes it, too
	strcat(shmfile, REG_DIR);
	strcat(shmfile, "/");
	sprintf(pid_str, "%d", reg->pid);
	strcat(shmfile, pid_str);
	if (unlink(shmfile) < 0) {
		printd(DBG_ERROR, "Could not unlink(%s): %s\n", shmfile,
				strerror(errno));
		goto fail;
	}
	memset(reg, 0, sizeof(struct registration));
	free(reg);
	reg = NULL;
	return 0;

fail:
	printd(DBG_ERROR, "failed\n");
	return -1;
}

int reg_lib_shutdown(void) {
	int err;
	if (lib_state.all_regs[0]) {
		err = reg_lib_disconnect(lib_state.all_regs[0]->id);
		if (err < 0) {
			printd(DBG_ERROR, "Could not disconnect reg %d\n",
					lib_state.all_regs[0]->id);
			goto fail;
		}
	}
	if (lib_state.all_regs)
		free(lib_state.all_regs);
	lib_state.all_regs = NULL;
	lib_state.valid = FALSE;
	return 0;

fail:
	return -1;
}

/*-------------------------------------- BACKEND FUNCTIONS --------------------*/

static int _pid_file_create(void) {
	int err;
	struct stat statbuf;
	memset(&statbuf, 0, sizeof(struct stat));

	// see if the pid file already exists
	err = stat(REG_PID, &statbuf);
	if (err < 0) {
		if (errno != ENOENT) { // okay if doesn't exist
			perror("stat on REG_PID");
			goto fail;
		}
	} else {
		fprintf(stderr, "It appears another backend is running."
				" If not, remove %s and retry.\n", REG_PID);
		goto fail;
	}

	// Make the directory. Perms are rwxrwx---
	err = mkdir(REG_DIR, S_IRWXU | S_IRWXG);
	if (err < 0) {
		if (errno != EEXIST) { // okay if already exists
			perror("mkdir(REG_DIR)");
			goto fail;
		}
	}

	// Create the PID file. Perms are rw-rw----
	// @todo TODO Put our PID into the PID file
	err = creat(REG_PID, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
	if (err < 0) {
		perror("creat(REG_PID)");
		goto fail;
	}

	return 0;

fail:
	return -1;
}

static int _pid_file_rm(void) {
	if (unlink(REG_PID) < 0) {
		perror("unlink(REG_PID)");
		goto fail;
	}
	return 0;
fail:
	return -1;
}

static regid_t _new_registration(const char *pid_str) {
	struct registration *reg = NULL;
	unsigned int idx = 0; //! index of new registration
	int fd = -1;

	if (!pid_str) {
		printd(DBG_ERROR, "NULL pid_str\n");
		return -1;
	}
	// find first NULL entry to put new registration into
	for ( ; idx < be_state.max_regs; idx++) {
		if (!be_state.all_regs[idx])
			break;
	}
	if (idx >= be_state.max_regs) {
		printd(DBG_ERROR, "Reached max regs\n");
		goto fail;
	}
	reg = calloc(1, sizeof(struct registration));
	if (!reg) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	reg->id = next_reg_id++;
	reg->pid = strtol(pid_str, NULL, 10);

	// Construct the filenames for the shm obj and reg file.
	memset(reg->shmfile, 0, sizeof(reg->shmfile));
	strcat(reg->shmfile, REG_OBJ_PREFIX);
	strcat(reg->shmfile, pid_str);
	memset(reg->regfile, 0, sizeof(reg->regfile));
	strcat(reg->regfile, REG_DIR);
	strcat(reg->regfile, "/");
	strcat(reg->regfile, pid_str);

	// 1. open the shm object
	fd = shm_open(reg->shmfile, O_RDWR, S_IRWXU | S_IRWXG);
	if (fd < 0) {
		printd(DBG_ERROR, "could not open shm obj '%s'\n", reg->shmfile);
		perror("shm_open");
		return -1;
	}
	reg->fd = fd;

	// 2. mmap the file to create the shared region
	reg->shm = mmap(NULL, SHM_SZ_B, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (reg->shm == MAP_FAILED) {
		printd(DBG_ERROR, "Could not mmap fd %d: %s\n", fd, strerror(errno));
		goto fail;
	}
	be_state.all_regs[idx] = reg; // an important step!
	printd(DBG_DEBUG, "id=%d pid=%d shm=%p fd=%d\n",
			reg->id, reg->pid, reg->shm, reg->fd);
	strcpy(reg->shm, "success");

	return reg->id;

fail:
	if (fd > -1)
		shm_unlink(reg->shmfile);
	if (reg)
		free(reg);
	return -1;
}

static int _rm_registration(const char *pid_str) {
	struct registration *reg = NULL;
	unsigned int idx = 0;
	pid_t pid; //! filename is the pid of the process unregistering
	int err;

	if (!pid_str) {
		printd(DBG_ERROR, "NULL pid_str\n");
		return -1;
	}
	// look for the registration struct, and deallocate it
	pid = strtol(pid_str, NULL, 10);
	for ( ; idx < be_state.max_regs; idx++) {
		reg = be_state.all_regs[idx];
		if (!reg)
			continue;
		if (pid == reg->pid)
			break;
	}
	if (!reg || idx >= be_state.max_regs) {
		printd(DBG_ERROR, "%s not registered\n", pid_str);
		goto fail;
	}
	be_state.all_regs[idx] = NULL; // take it out
	err = munmap(reg->shm, SHM_SZ_B);
	if (err < 0) {
		printd(DBG_ERROR, "munmap of shm failed: %s\n", strerror(errno));
		goto fail;
	}
	err = shm_unlink(reg->shmfile);
	if (err < 0) {
		printd(DBG_ERROR, "unlink of shm obj failed: %s\n", strerror(errno));
		goto fail;
	}
	printd(DBG_DEBUG, "id=%d\n", reg->id);
	memset(reg, 0, sizeof(struct registration));
	free(reg);
	reg = NULL;
	return 0;

fail:
	printd(DBG_ERROR, "failed\n");
	return -1;
}

static regid_t _find_regid(const char *pid_str) {
	struct registration *reg = NULL;
	unsigned int idx;
	pid_t pid;
	regid_t id = -1;
	if (!pid_str) {
		printd(DBG_ERROR, "NULL pid_str\n");
		return -1;
	}
	pid = strtol(pid_str, NULL, 10);
	for (idx = 0; idx < be_state.max_regs; idx++) {
		reg = be_state.all_regs[idx];
		if (!reg)
			continue;
		if (pid == reg->pid) {
			id = reg->id;
			break;
		}
	}
	return id;
}

static void _inotify_cleanup(void *arg) {
	printd(DBG_INFO, "Callback thread terminating\n");
	be_state.inotify.alive = FALSE;
	// @todo TODO notify someone we died, could have been due to an
	// unrecoverable error
}

static void *_inotify_thread(void *arg) {
	int fd = -1, wd = -1;	//! inotify instance; watch descriptor
	int len; //! num bytes returned from read on inotify fd
	char *events = NULL;	//! array of inotify events
	struct inotify_event *event; //! current event we're processing
	int err;

	// cleanup function will always be called when thread is killed or exits
	pthread_cleanup_push(_inotify_cleanup, NULL);
	be_state.inotify.alive = TRUE;
	printd(DBG_INFO, "Callback thread is alive\n");

	/*
	 * Source: www.linuxjournal.com/article/8478
	 */
	events = calloc(1, INOTIFY_BUFFER_SZ);
	if (!events) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	fd = inotify_init();
	if (fd < 0) {
		printd(DBG_ERROR, "notify_init failed: %s\n", strerror(errno));
		pthread_exit(NULL);
	}
	wd = inotify_add_watch(fd, REG_DIR, IN_CREATE | IN_DELETE);
	if (wd < 0) {
		printd(DBG_ERROR, "inotify_add_watch failed: %s\n", strerror(errno));
		goto fail;
	}
	while (1) {
		printd(DBG_DEBUG, "waiting\n");
		len = read(fd, events, INOTIFY_BUFFER_SZ); // sleep until next event
		// check length
		if (len < 0) {
			if (errno == EINTR) {
				printd(DBG_WARNING, "inotify read interrupted by signal\n");
				continue; // re-issue read()
			} else {
				perror("read on inotify fd");
				break;
			}
		} else if (len == 0) {
			printd(DBG_ERROR, "inotify read 0 bytes\n");
			break;
		}
		// length is okay, loop through all events
		int i = 0;
		while (i < len) {
			event = (struct inotify_event *)(&events[i]);
			// nameless event... since we trigger ONLY on create/delete this can
			// signify the directory itself was deleted or something else bad
			if (event->len == 0) {
				printd(DBG_WARNING, "bad inotify: wd=%d mask=0x%x, cookie=0x%x"
						" len=0\n", event->wd, event->mask, event->cookie);
				break;
			}
			printd(DBG_DEBUG, "inotify: wd=%d mask=0x%x, cookie=0x%x len=%u"
					" name='%s'\n", event->wd, event->mask, event->cookie,
					event->len, event->name);
			// double-check the wd is ours...
			if (event->wd != wd) {
				printd(DBG_WARNING, "wd doesn't match\n");
				break;
			}
			regid_t regid;
			switch (event->mask) {
				case IN_CREATE:
					regid = _new_registration(event->name);
					if (regid < 0) {
						printd(DBG_ERROR, "_new_registration failed\n");
						goto fail;
					}
					be_state.inotify.func(CALLBACK_NEW, regid);
					break;
				case IN_DELETE:
					regid = _find_regid(event->name);
					if (regid < 0)
						printd(DBG_WARNING, "%s not registered\n", event->name);
					else {
						be_state.inotify.func(CALLBACK_DEL, regid);
						err = _rm_registration(event->name);
						if (err < 0) {
							printd(DBG_ERROR, "Could not remove reg %d\n", regid);
							break;
						}
					}
					break;
				default:
					printd(DBG_ERROR, "inotify: unexpected event mask\n");
					continue;
			}
			i += INOTIFY_EVENT_SZ + event->len;
		}
	}

fail:
	if (fd > -1)
		close(fd); // removes watches, cleans up inotify instance
	fd = -1;
	// @todo TODO Clear each individual registration struct, then array
	if (!events)
		free(events);
	events = NULL;

	pthread_cleanup_pop(1);
	assert(0); // not reachable
	__builtin_unreachable();
}

int reg_be_init(unsigned int max_regs) {
	int err;

	if (be_state.valid == TRUE) {
		printd(DBG_WARNING, "Backend has already initialized registration\n");
		goto fail;
	}

	err = _pid_file_create();
	if (err < 0) {
		printd(DBG_ERROR, "PID create failed\n");
		goto fail;
	}

	if (max_regs == 0) {
		printd(DBG_ERROR, "max regs is zero\n");
		goto fail;
	}

	be_state.all_regs = calloc(max_regs, sizeof(struct registration *));
	if (!be_state.all_regs) {
		printd(DBG_ERROR, "Out of memory\n");
		fprintf(stderr, "Out of memory\n");
		goto fail;
	}
	be_state.max_regs = max_regs;

	printd(DBG_INFO, "SHM size %d MiB\n", SHM_SZ_MB);
	printd(DBG_INFO, "Reg dir '%s'\n", REG_DIR);
	printd(DBG_INFO, "Backend PID '%s'\n", REG_PID);

	printd(DBG_INFO, "libregistration initialized\n");

	be_state.valid = TRUE;
	return 0;

fail:
	return -1;
}

int reg_be_shutdown(void) {
	int err;

	if (be_state.valid == FALSE) {
		printd(DBG_WARNING, "init never called\n");
		return 0;
	}

	// terminate inotify thread
	if (be_state.inotify.alive == TRUE) {
		err = pthread_cancel(be_state.inotify.tid);
		if (err != 0) {
			printd(DBG_ERROR, "Could not cancel inotify thread\n");
		}
		err = pthread_join(be_state.inotify.tid, NULL);
		if (err != 0) {
			printd(DBG_ERROR, "Could not join inotify thread\n");
		}
		// the OS will clean up any errors here, so keep going :)
	}

	// TODO Free memory allocated to reg elements and then the array
	be_state.valid = FALSE;
	free(be_state.all_regs);
	memset(&be_state, 0, sizeof(struct backend_state));

	err = _pid_file_rm();
	if (err < 0) { // user will have to manually remove the file
		printd(DBG_ERROR, "PID file removal failed\n");
	}

	// TODO check to see if any open mmap'd files exist, and what to do if so

	printd(DBG_INFO, "libregistration shutdown\n");
	return 0;
}

int reg_be_callback(void (*callback)(enum callback_event e, regid_t id)) {
	int err;
	if (!callback) {
		printd(DBG_ERROR, "NULL argument\n");
		goto fail;
	}
	printd(DBG_INFO, "Creating inotify thread\n");
	// @todo TODO Block all signals for this thread.
	be_state.inotify.func = callback;
	err = pthread_create(&be_state.inotify.tid, NULL, _inotify_thread, NULL);
	if (err < 0) {
		printd(DBG_ERROR, "Could not create inotify thread\n");
		goto fail;
	}
	return 0;

fail:
	return -1;
}

void* reg_be_get_shm(regid_t id) {
	void *shm = NULL;
	if (id < LOWEST_VALID_REGID) {
		printd(DBG_DEBUG, "invalid id %d\n", id);
		return NULL;
	}
	// TODO search for id
	if (be_state.valid && be_state.all_regs[0])
		shm = be_state.all_regs[0]->shm;
	return shm;
}

size_t reg_be_get_shm_size(regid_t id) {
	size_t size = -1;
	if (id < LOWEST_VALID_REGID) {
		printd(DBG_DEBUG, "invalid id %d\n", id);
		return 0;
	}
	// TODO search for id
	if (be_state.valid && be_state.all_regs[0])
		size = SHM_SZ_B;
	return size;
}

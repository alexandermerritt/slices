/**
 * @file runtime.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-25
 *
 * @brief Entry point for applications to enter assembly runtime. Watch for
 * applications to register, accept shared memory locations and assign them
 * assemblies.
 *
 * FIXME This file is a mess. Clean it up later to keep track of all sinks it
 * spawns. And find a reasonable way to specify assembly hint structures/files.
 */

#define _GNU_SOURCE
#include <sched.h>

// System includes
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// Project includes
#include <assembly.h>
#include <debug.h>
#include <mq.h>
#include <util/timer.h>

// Directory-immediate includes
#include "sinks.h"

//#define VARIABLE_BATCHING

#ifdef VARIABLE_BATCHING
// Temporary code for varying the batch size. The idea is to allow
// an application to run multiple iterations, over time varying the
// batch size to observe the performance impact.
static size_t bsize = 1; //! Current batch size.
static size_t incr_on_mod = 1; //! # app entries between batch size increments
static size_t count = 0; //! internal counter of app entries for use with incr
#endif	/* VARIABLE_BATCHING */

/*-------------------------------------- EXTERNAL VARIABLES ------------------*/

/*-------------------------------------- INTERNAL STATE ----------------------*/

/* an application process using shadowfax */
struct app
{
    struct list_head link;
    struct mq_state mq;
    asmid_t asmid;
    pid_t pid;
    unsigned long pid_group; /* e.g. MPI ranks belonging to same app */
};
static LIST_HEAD(apps);
#define for_each_app(app, apps) \
    list_for_each_entry(app, &apps, link)
static pthread_mutex_t apps_lock = PTHREAD_MUTEX_INITIALIZER;

// Daemon state
#ifndef NO_DAEMONIZE
static pid_t sid;
static FILE *logf = NULL;
#endif
static char *log; /** determined based on machine name */
static const char wd[] = "."; //! working dir of runtime once exec'd

/*-------------------------------------- INTERNAL FUNCTIONS ------------------*/

static void msg_received(msg_event e, pid_t pid, void *data)
{
    int err;
    struct app *app = NULL;
    asmid_t asmid;
    assembly_key_uuid key;

    switch ( e )
    {

    case ATTACH_CONNECT:
    {
        printd(DBG_INFO, "msg ATTACH_CONNECT PID %d\n", pid);
        app = calloc(1, sizeof(*app));
        if ( app )
        {
            INIT_LIST_HEAD(&app->link);
            app->pid = pid;

            pthread_mutex_lock(&apps_lock);
            list_add(&app->link, &apps);
            pthread_mutex_unlock(&apps_lock);

            err = attach_allow(&app->mq, pid); /* open app's mqueue */
            if (err < 0)
                fprintf(stderr, "Error allowing attachment to PID %d\n", pid);

            err = attach_send_allow(&app->mq, true);
            if (err < 0)
                fprintf(stderr, "Error notifying PID %d of attachment\n", pid);
        }
        else
            fprintf(stderr, "Out of memory\n");
    }
    break;

    case ATTACH_DISCONNECT: /* process is going to die */
    {
        printd(DBG_INFO, "msg ATTACH_DISCONNECT PID %d\n", pid);
        pthread_mutex_lock(&apps_lock);
        for_each_app(app, apps)
            if ( app->pid == pid )
                break;
        if ( app )
            list_del(&app->link);
        pthread_mutex_unlock(&apps_lock);
        if ( app )
        {
            printd(DBG_INFO, "disconnect: found app state for PID %d\n", pid);
            if (0 > assembly_teardown(app->asmid))
            {
                printd(DBG_ERROR, "fail assembly_teardown asmid %lu\n",
                        app->asmid);
            }
            free(app);
        }
        else
            fprintf(stderr, "No PID %d found, yet termination msg received\n",
                    pid);
    }
    break;

    case ATTACH_REQUEST_ASSEMBLY:
    {
        printd(DBG_INFO, "msg ATTACH_REQUEST_ASSEMBLY PID %d\n", pid);
        char uuid_str[64];
        struct assembly_hint *hint = data;

        asmid = assembly_request(hint);
        assembly_print(asmid);

        /* hand off to interposer for mapping */
        err = assembly_export(asmid, key); /* modifies 'key' */
        if (err < 0) {
            fprintf(stderr, "Error exporting assm\n");
            break;
        }

        uuid_unparse(key, uuid_str);
        printd(DBG_INFO, "exported key '%s'\n", uuid_str);

        /* add application to our list to track */
        pthread_mutex_lock(&apps_lock);
        for_each_app(app, apps)
            if ( app->pid == pid )
                break;
        pthread_mutex_unlock(&apps_lock);
        if (app) {
            app->asmid = asmid;
            err = attach_send_assembly(&app->mq, key);
            if (err < 0) {
                fprintf(stderr, "Error sending assm to PID %d\n", pid);
            }
        }
        else
            fprintf(stderr, "No PID %d found, yet assm req msg received\n",
                    pid);
    }
    break;

    default:
    {
        fprintf(stderr, "Unknown or unexpected message received:"
                " pid %d event %d\n", pid, e);
    }
    break;
    }
}

static void sigint_handler(int sig)
{
	; // Do nothing, just prevent it from killing us
}

static int start_runtime(enum node_type type, const char *main_ip)
{
	int err;
    err = attach_open(msg_received);
    if ( err < 0 ) {
        fprintf(stderr, "Could not open attach interface\n");
        return -1;
    }
	err = assembly_runtime_init(type, main_ip);
	if (err < 0) {
		fprintf(stderr, "Could not initialize assembly runtime\n");
		return -1;
	}
	return 0;
}

static void shutdown_runtime(void)
{
	int err;
    /* TODO verify all PIDs have exited */
	err = attach_close();
	if (err < 0) {
		fprintf(stderr, "Could not close attach interface\n");
	}
	err = assembly_runtime_shutdown();
	if (err < 0) {
		fprintf(stderr, "Could not shutdown assembly runtime\n");
	}
    if (err >= 0)
        printf("\nAssembly runtime shut down.\n");
}

// TODO use getopt for all the compile flags
static bool verify_args(int argc, char *argv[], enum node_type *type)
{
	const char main_str[] = "main";
	const char minion_str[] = "minion";
	if (!argv)
		return false;
	if (argc < 2 || argc > 3)
		return false;
	if (argc == 2) { // ./runtime main
		if (strncmp(argv[1], main_str, strlen(main_str)) != 0)
			return false;
		*type = NODE_TYPE_MAIN;
	} else if (argc == 3) { // ./runtime minion <ip-addr>
		if (strncmp(argv[1], minion_str, strlen(minion_str)) != 0)
			return false;
		// TODO verify ip via regex
		*type = NODE_TYPE_MINION;
	}
	return true;
}

static void print_usage(void)
{
	const char usage_str[] =
		"Usage: ./runtime main\n"
		"       ./runtime minion <ip-addr>\n";
	fprintf(stderr, usage_str);
}

static int daemonize(void)
{
	struct sigaction action;
#ifndef NO_DAEMONIZE
	int logfd = -1;

	printf(">:[ daemonizing ... log file at %s/%s\n", wd, log);

	pid_t pid = fork();
	if (pid < 0)
		return -(errno);
	if (pid > 0)
		_exit(0);

	umask(0);
	chdir(wd);

	if (0 > (logf = fopen(log, "w")))
		return -(errno);
	logfd = fileno(logf);
	if (0 > dup2(logfd, 0))
		return -(errno);
	if (0 > dup2(logfd, 1))
		return -(errno);

	if (0 > (sid = setsid()))
		return -(errno);
	printf("Send SIGINT to pid %d to terminate daemon.\n", getpid());
#endif	/* !NO_DAEMONIZE */

	memset(&action, 0, sizeof(action));
	action.sa_handler = sigint_handler;
	sigemptyset(&action.sa_mask);
	if (0 > sigaction(SIGINT, &action, NULL))
		return -(errno);

	return 0;
}

/*-------------------------------------- ENTRY -------------------------------*/

int main(int argc, char *argv[])
{
	int err = 0;
	sigset_t mask;
	enum node_type type = NODE_TYPE_INVALID;

#ifdef DEBUG
	printf("(Built with debug symbols)\n");
#endif

	if (!verify_args(argc, argv, &type)) {
		print_usage();
		return -1;
	}

	/** configure log file name */
	log = calloc(1, HOST_NAME_MAX << 1);
	if (!log) {
		fprintf(stderr, "No memory left\n");
		exit(1);
	}
	strcat(log, "runtime-");
	gethostname((log + strlen(log)), HOST_NAME_MAX);
	strcat(log, ".log");

	if (0 > daemonize())
		return -1;

	if (argc == 2)
		err = start_runtime(type, NULL);
	else if (argc == 3)
		err = start_runtime(type, argv[2]);
	if (err < 0) {
		fprintf(stderr, "Could not initialize. Check your arguments.\n");
		return -1;
	}

	// Wait for any signal.
	sigemptyset(&mask);
	sigsuspend(&mask);

	shutdown_runtime();
	return 0;
}

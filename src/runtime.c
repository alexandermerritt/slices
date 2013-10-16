/**
 * @file runtime.c
 * @author Alexander Merritt, merritt.alex@gatech.edu
 */

#define _GNU_SOURCE
#include <sched.h>

// System includes
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <stdbool.h>
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

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

typedef char dns_t[HOST_NAME_MAX];

// Configuration flags
struct flags {
    int is_master, use_pbs;
    int quiet; // hush stdout and stderr
    int nohup; // catch SIGHUP
};

// Global state
struct glob {
    struct flags flags;
    int pbs_num_nodes;  // valid if --pbs --master
    dns_t *pbs_nodes;   // valid if --pbs --master
    dns_t masterDNS;    // valid if --minion
    struct list_head apps;
    pthread_mutex_t apps_lock;
};

// Connecting process
struct app
{
    struct list_head link;
    struct mq_state mq;
    asmid_t asmid;
    pid_t pid;
    unsigned long pid_group; /* e.g. MPI ranks belonging to same app */
};

#define for_each_app(app, apps) \
    list_for_each_entry(app, &apps, link)

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

static struct glob glob;
static int sigints_received = 0;
static const struct option options[] = {
    {"pbs", no_argument, (int*)&glob.flags.use_pbs, true},
    {"quiet", no_argument, (int*)&glob.flags.quiet, true},
    {"nohup", no_argument, (int*)&glob.flags.nohup, true},
    {"master", no_argument, (int*)&glob.flags.is_master, true},
    {"minion", required_argument, NULL, 'n'},
    {NULL, no_argument, NULL, 0} // terminator
};

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

static void msg_received(msg_event e, pid_t pid, void *data)
{
    int err;
    struct app *app = NULL;
    asmid_t asmid;
    assembly_key key;

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

            pthread_mutex_lock(&glob.apps_lock);
            list_add(&app->link, &glob.apps);
            pthread_mutex_unlock(&glob.apps_lock);

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
        pthread_mutex_lock(&glob.apps_lock);
        for_each_app(app, glob.apps)
            if ( app->pid == pid )
                break;
        if ( app )
            list_del(&app->link);
        pthread_mutex_unlock(&glob.apps_lock);
        if ( app )
        {
            printd(DBG_INFO, "disconnect: found app state for PID %d\n", pid);
            if (0 > assembly_teardown(app->asmid))
            {
                printd(DBG_ERROR, "fail assembly_teardown asmid %lu\n",
                        app->asmid);
            }

            /* close app's mqueue so it can unlink */
            err = attach_dismiss(&app->mq);
            if (err < 0)
                fprintf(stderr, "Error dismissing attachment to PID %d\n", pid);

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
        struct assembly_hint *hint = data;

        asmid = assembly_request(hint, pid);

        /* hand off to interposer for mapping */
        err = assembly_export(asmid, &key);
        if (err < 0) {
            fprintf(stderr, "Error exporting assm\n");
            break;
        }

        printd(DBG_INFO, "exported key %d\n", key);

        /* add application to our list to track */
        pthread_mutex_lock(&glob.apps_lock);
        for_each_app(app, glob.apps)
            if ( app->pid == pid )
                break;
        pthread_mutex_unlock(&glob.apps_lock);
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

// Opens PBS_NODEFILE to extract the unique node names.
// Called only by --master when --pbs
static int parse_pbs_nodes(void)
{
    int idx, ret = -1;
    FILE *file = NULL;
    char *line = NULL;
    const char *nodefile = NULL;
    const int line_len = HOST_NAME_MAX;

    BUG(!glob.flags.is_master || !glob.flags.use_pbs);

    if (!getenv("PBS_ENVIRONMENT")) {
        fprintf(stderr, ">> Error: PBS environment variables don't seem to exist\n");
        return -1;
    }
    BUG(!getenv("PBS_NUM_NODES"));
    glob.pbs_num_nodes = atoi(getenv("PBS_NUM_NODES"));
    BUG(glob.pbs_num_nodes < 1);

    nodefile = getenv("PBS_NODEFILE");
    BUG(!nodefile);

    if (!(file = fopen(nodefile, "r")))
        goto out;
    if (!(line = calloc(1, line_len)))
        goto out;
    if (!(glob.pbs_nodes = calloc(glob.pbs_num_nodes, sizeof(*glob.pbs_nodes))))
        goto out;

    if (gethostname(glob.pbs_nodes[0], HOST_NAME_MAX))
        goto out;
    *strchrnul(glob.pbs_nodes[0], '.') = '\0'; // make into short DNS
    idx = 1;

    char *prev = glob.pbs_nodes[0];
    while (fgets(line, line_len, file)) {
        if (!prev || !strstr(line, prev)) {
            prev = glob.pbs_nodes[idx++];
            sscanf(line, "%s\n", prev);
        }
    }
    ret = 0;

out:
    if (ret) {
        fprintf(stderr, ">> Error: %s: %s\n", __func__, strerror(errno));
        if (glob.pbs_nodes)
            free(glob.pbs_nodes);
        glob.pbs_nodes = NULL;
    }
    if (file)
        fclose(file);
    if (line)
        free(line);
    return ret;
}

static int parse_args(int argc, char *argv[])
{
    int opt, idx;
    while (-1 != (opt = getopt_long(argc, argv, "", options, &idx)))
        if (opt == 'n')
            strncpy(glob.masterDNS, optarg, HOST_NAME_MAX);
    if (glob.flags.is_master && glob.flags.use_pbs)
        if (parse_pbs_nodes())
            return -1;
    return 0;
}

static void print_usage(void)
{
	const char usage_str[] =
		"Usage: ./shadowfax --master [--pbs] [FLAGS]..\n"
        "       ./shadowfax --minion=masterDNS [FLAGS]..\n"
        "--quiet    Close stdout\n"
        "--nohup    catch/ignore SIGHUP\n";
	fprintf(stderr, usage_str);
}

static void sighandler(int sig)
{
    printd(DBG_DEBUG, "sig %s (%d) caught\n", strsignal(sig), sig);
    if (sig == SIGINT)
        sigints_received++;
}

static int setsignals(void)
{
	struct sigaction action;

	memset(&action, 0, sizeof(action));
	action.sa_handler = sighandler;
	sigemptyset(&action.sa_mask);
	if (0 > sigaction(SIGINT, &action, NULL))
		return -(errno);
    if (glob.flags.nohup) {
        if (0 > sigaction(SIGHUP, &action, NULL))
            return -(errno);
    }

	return 0;
}

// if master and use_pbs then auto-launch minions
static int start_others(void)
{
    int n, err;
    const int cmdlen = 256;
    char cmd[cmdlen];
    const char ssh[] = "ssh -oStrictHostKeyChecking=no -q -f";
    for (n = 1; n < glob.pbs_num_nodes; n++) {
        memset(cmd, 0, sizeof(*cmd) * cmdlen);
        snprintf(cmd, cmdlen, "%s %s " // ssh node ..
                "\"source sfmodules; shadowfax --minion=%s --quiet --nohup\"",
                ssh, glob.pbs_nodes[n], glob.pbs_nodes[0]);
        printd("%s\n", cmd);
        err = system(cmd);
        if (err) {
            fprintf(stderr, ">> Error launching on %s\n", glob.pbs_nodes[n]);
            if (n > 1)
                fprintf(stderr, ">> Other instances exist; kill them manually\n");
            return -1;
        }
    }
    return 0;
}

#if 0 // works but kids sends us SIGSTOP when shadowfax is &'d
static int stop_others(void)
{
    int n;
    const int cmdlen = 256;
    char cmd[cmdlen];
    const char ssh[] = "ssh -oStrictHostKeyChecking=no";
    for (n = 1; n < glob.pbs_num_nodes; n++) {
        memset(cmd, 0, sizeof(*cmd) * cmdlen);
        snprintf(cmd, cmdlen, "%s %s " // ssh node ..
                "\"killall -s SIGINT shadowfax\"",
                ssh, glob.pbs_nodes[n]);
        printd("%s\n", cmd);
        if (-1 == system(cmd))
            fprintf(stderr, ">> Error stopping instance on %s\n",
                    glob.pbs_nodes[n]);
    }
    // give them time to disconnect
    if (glob.pbs_num_nodes > 1)
        sleep(glob.pbs_num_nodes / 4 + 4);
    return 0;
}
#endif

static int start_runtime(void)
{
    if (attach_open(msg_received))
        return -1;
    if (!glob.flags.is_master) {
        if (assembly_runtime_init(NODE_TYPE_MINION, glob.masterDNS))
            return -1;
    } else {
        if (assembly_runtime_init(NODE_TYPE_MAIN, NULL))
            return -1;
        if (glob.flags.use_pbs && start_others())
            return -1;
    }
	return 0;
}

static void shutdown_runtime(void)
{
    /* TODO verify all PIDs have exited */
	if (attach_close())
		fprintf(stderr, ">> Error closing MQ\n");
#if 0
    if (glob.flags.is_master && stop_others())
        fprintf(stderr, ">> Error stopping other instances\n");
#endif
	if (assembly_runtime_shutdown())
		fprintf(stderr, ">> Erro shutting down runtime\n");
}

static int make_quiet(void)
{
    int fd = open("/dev/null", O_APPEND);
    if (0 > fd)
        return -1;
    if (0 > dup2(fd, 1))
        return -1;
    if (0 > dup2(fd, 2))
        return -1;
    return 0;
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char *argv[])
{
	sigset_t mask;

    memset(&glob, 0, sizeof(glob));
    INIT_LIST_HEAD(&glob.apps);
    pthread_mutex_init(&glob.apps_lock, NULL);

	if (parse_args(argc, argv))
		return -1;
    if (!glob.flags.is_master && '\0' == *glob.masterDNS) {
        print_usage();
        exit(EXIT_FAILURE);
    }
    if (glob.flags.quiet && make_quiet())
        exit(EXIT_FAILURE);

    if (0 > attach_clean()) {
        fprintf(stderr, ">> Error cleaning old message queues\n");
        exit(EXIT_FAILURE);
    }
	if (0 > setsignals()) {
        fprintf(stderr, ">> Error initializing signals\n");
        exit(EXIT_FAILURE);
    }
    if (start_runtime()) {
        fprintf(stderr, ">> Error starting runtime\n");
        exit(EXIT_FAILURE);
    }

	sigemptyset(&mask);
    while (sigints_received == 0)
        sigsuspend(&mask);

	shutdown_runtime();
    attach_clean();
    exit(EXIT_SUCCESS);
}

/**
 * @file backend/assembly/remote.h
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-25
 * @brief Header defining functions to enable/configure remote end of data path.
 */

#ifndef _REMOTE_H
#define _REMOTE_H

/*-------------------------------------- DEFINITIONS -------------------------*/

#define REMOTE_CUDA_PORT	"46214"

/**
 * Signal the remote sink should wait for from the parent runtime for
 * termination.
 */
#define REMOTE_TERM_SIG		SIGTERM

/**
 * argv[0] of the exec'd sink, so we know it was created by us and not
 * executedon the command line (a crude method)
 */
#define REMOTE_EXEC_NAME	"remotesink-a65d7509-3481-4569-946e-9022e3ff1ea3"

#endif	/* _REMOTE_H */

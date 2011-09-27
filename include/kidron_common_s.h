/**
 * @file kidron_common_s.h
 * @brief The common data structures for the system
 *
 * @date Sep 13, 2011
 * @author Magda Slawinska, magg dot gatech __at_ gmail dot com
 *
 */

#ifndef KIDRON_COMMON_S_H_
#define KIDRON_COMMON_S_H_

#include <glib.h>

/**
 * The structure that reflects fields in the ini file
 */
typedef struct _ini {

	//! the name of the ini file
	char*		ini_name;
	// [network] stuff
	//! the host where the backend is running
	gchar*		backend_host;
	//! the host where the VC_Manager is running
	gchar*		vc_manager_host;

	// [interposer]
	//! tells if the interposer library functions need to be called locally
	//! or remotely; if local set to true, then they will be called
	//! locally, if false then they will be called remotely
	gboolean	local;

} ini_t;

#endif /* KIDRON_COMMON_S_H_ */

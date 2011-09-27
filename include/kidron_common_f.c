/**
 * @file kidron_common_f.c
 * @brief This contains common functions used by the backend and the interposer
 * library
 *
 * @date Sep 13, 2011
 * @author Magda Slawinska, magg dot gatech __at_ gmail dot com
 */

#include "kidron_common_s.h"

#include <glib.h>
#include "debug.h"
#include <unistd.h>
#include <string.h>


/**
 * The idea of how to work with ini file:
 *
 * 1. parse the command line
 * 2. ini_getIni
 * 3. when you are done: ini_freeIni
 */

/**
 * Gets the ini file and transforms it to the structure
 *
 * @param pIni; it is expected that the name pIni->ini_name is provided
 *
 * @return 0 if everything went smoothly, otherwise exits
 */
int ini_getIni(ini_t* pIni){
	GKeyFile* key_file = NULL;
	GError* error = NULL;
	key_file = g_key_file_new();

	ptr_null_exit(pIni, "Pointer to ini file is NULL. Provide the correct pointer");
	ptr_null_exit(pIni->ini_name, "The name of the ini file is null. Provide the correct one");

	if (!g_key_file_load_from_file(key_file, pIni->ini_name, G_KEY_FILE_NONE, &error))
		p_exit(" Can't parse the key file: %s. Quitting ...\n", error->message);

	pIni->backend_host = g_key_file_get_string(key_file, "network", "backend_host", &error);
	if( NULL == pIni->backend_host&& (G_KEY_FILE_ERROR_KEY_NOT_FOUND == error->code ||
			G_KEY_FILE_ERROR_GROUP_NOT_FOUND == error->code) )
		p_exit("Can't read the value from ini file : [%s:%s]\n", "network", "backend_host");

	pIni->vc_manager_host = g_key_file_get_string(key_file, "network", "vc_manager_host", &error);
	if( NULL == pIni->backend_host&& (G_KEY_FILE_ERROR_KEY_NOT_FOUND == error->code ||
				G_KEY_FILE_ERROR_GROUP_NOT_FOUND == error->code) )
			p_exit("Can't read the value from ini file : [%s:%s]\n", "network", "backend_host");

	pIni->local = g_key_file_get_boolean(key_file, "interposer", "local", &error);
	if( FALSE == pIni->local && NULL != error && (G_KEY_FILE_ERROR_KEY_NOT_FOUND == error->code ||
			G_KEY_FILE_ERROR_INVALID_VALUE == error->code ))
		p_exit("Can't read the value from ini file: [%s:%s]\n", "interposer", "local");


	g_key_file_free(key_file);

	return 0;
}

/**
 * frees resources allocated for the ini_t structure
 * @param pIni The pointer to the structure
 * @return 0
 */
int ini_freeIni(ini_t* pIni){

	g_free(pIni->backend_host);
	pIni->backend_host = NULL;

	g_free(pIni->vc_manager_host);
	pIni->vc_manager_host = NULL;

	return 0;
}

/**
* Reads the network:local value from the ini file and returns
* the numerical value
*
* @param pIni The pointer to the pIni structure
*
* @return 1 - Local GPU will be invoked (means interposer:local is yes)
*         0 - remote GPU will be invoked (means interposer:local is set no)
*/
int ini_getLocal(const ini_t* pIni){

	if( TRUE == pIni->local  ){
		p_info("Local GPU will be invoked\n");
		return 1;
	} else {
		p_info("Remote GPU will be invoked\n");
		return 0;
	}
}

/**
 * returns the string of the backend host read from the ini file
 * @param pIni The structure containing the fields from the ini file
 * @return the string representing the backend hostname
 */
char* ini_getBackendHost(const ini_t* pIni){
	ptr_null_exit(pIni, "The pIni pointer is NULL. Quitting .... \n");
	ptr_null_exit(pIni->backend_host, "The backend hostname is NULL. Quitting ...\n");

	return pIni->backend_host;
}


/**
 * sets the hostname to the ini file; exits if there are any problems
 * @param pIni The pointer the ini structure that contains the
 *             ini file name
 */
void ini_vc_setVCManagerHost(const ini_t* pIni){
	GKeyFile*		key_file;
	GKeyFileFlags	flags = G_KEY_FILE_KEEP_COMMENTS | G_KEY_FILE_KEEP_TRANSLATIONS;
	GError*			error = NULL;
	char 			hostname[256];

	if( gethostname(hostname, sizeof(hostname)) != 0 )
		p_exit("Problems with getting the host name\n");

	key_file = g_key_file_new ();
	ptr_null_exit(key_file, "Problems with openin he ini file\n");

	if( !g_key_file_load_from_file(key_file, pIni->ini_name, flags, &error)){
		p_exit("Problems with loading the file: %s (code: %d)\n",
				error->message, error->code);
	}

	g_key_file_set_string(key_file, "network", "vc_manager_host", hostname);

	gsize length;
	gchar* ini_content;

	ini_content = g_key_file_to_data(key_file, &length, &error);
	FILE * f;
	f = fopen(pIni->ini_name, "w");
	fwrite(ini_content, 1, strlen(ini_content), f);

	fclose(f);
	g_key_file_free(key_file);

	p_debug("[network:vc_manager_host] set to: %s\n", hostname);
}


/**
 * parse the command line options; the version for the
 *
 * @param argc the argument count as provided by the command line
 * @param argv the arguments as provided by the command line
 * @param pIni The pointer to the ini structure; sets the ini_name file for the
 *             structure pIni
 */
void
ini_cmd_parse(int argc, char** argv, ini_t* pIni){
	GError* error = NULL;
	GOptionContext* context;

	GOptionEntry entries[] = {
	  { "ini-file", 'f', 0, G_OPTION_ARG_FILENAME, &pIni->ini_name,
			  "The name of or path to the configuration ini file", "file_name.ini" },
	  { NULL, '\0', 0, 0,  NULL, NULL, NULL}
	};

	context = g_option_context_new("- Kidron Virtual Clusters Utils");
	g_option_context_set_summary(context,
	"The software is a part of the Kidron system and implements the concept of Virtual Clusters. "
	);
	g_option_context_add_main_entries( context, entries, NULL);

	if( !g_option_context_parse(context, &argc, &argv, &error)){
		p_exit("Option parsing failed: %s\n", error->message);
	}

	// check if the configuration file exists
	if (access(pIni->ini_name, F_OK) == -1) {
		p_exit("File %s doesn't exists\n", pIni->ini_name);
	}

	// clean things if possible
	g_option_context_free(context);
}

/**
 * prints the ini file
 * @param pIni The structure to be printed
 */
void
ini_print(const ini_t* pIni){
	printf("backend_host_name = %s\n", pIni->backend_host);
	printf("backend config file name = %s\n", pIni->ini_name);
	printf("backend vc_manager_host = %s\n", pIni->vc_manager_host);
}

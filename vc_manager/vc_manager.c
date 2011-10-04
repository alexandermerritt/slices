/**
 * @file vc_manager.c
 *
 * @date Sep 26, 2011
 * @author Magda Slawinska, aka Magic Magg, magg dot gatech at gmail dot edu
 *
 * The process responsible for maintaining information about
 * resources available for Virtual Clusters; it allows to interact
 * with backends.
 */

#include <pthread.h>
#include "connection.h"

#include "kidron_common_s.h"  // ini_t
#include "debug.h"

// kidron_common_f.c
extern void ini_cmd_parse(int argc, char** argv, ini_t* pIni);
extern int ini_getIni(ini_t* pIni);
extern int ini_freeIni(ini_t* pIni);
extern void ini_vc_setVCManagerHost(const ini_t* pIni);

/**
 * the loop for the vc_mgr_thread
 */
void* vc_mgr_main(){
	p_not_impl();
	return NULL;
}

/**
 * it expects the argument -f ini_file_name
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv){
	pthread_t vc_mgr_thread;

	// stores diagnostic information
	int ret_code;

	ini_t	ini;

	ini_cmd_parse(argc, argv, &ini);
	ini_vc_setVCManagerHost(&ini);
	ini_getIni(&ini);

	ret_code = pthread_create(&vc_mgr_thread, NULL, &vc_mgr_main, NULL);
	pth_exit(ret_code);

	pthread_join(vc_mgr_thread, NULL);

	p_debug("server thread says you bye bye!\n");
	ini_freeIni(&ini);
	return 0;
}

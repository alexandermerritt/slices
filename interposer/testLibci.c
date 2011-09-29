/**
 * @file testLibci.c
 * @brief
 *
 * @date Mar 10, 2011
 * @author Magda Slawinska, magg __at_ gatech __dot_ edu
 */

#include "packetheader.h"
#include "debug.h"	// ERROR, OK
#include <pthread.h>

#include <glib.h>

extern int l_setMetThrReq(cuda_packet_t ** const pPacket, const uint16_t methodId);


/**
 * tests if the packet is correctly initialized
 */
void test_l_setMetThrReq(void){
	cuda_packet_t packet;
	cuda_packet_t * pPacket = &packet;
	uint16_t methodId = 10;

	// check if we really can set the content of the packet
	g_assert( OK == l_setMetThrReq(&pPacket, methodId));
	// now check the values
	g_assert( methodId == pPacket->method_id );
	g_assert( methodId == packet.method_id );
	g_assert( pthread_self() == pPacket->thr_id);
	g_assert( pthread_self() == packet.thr_id);
	g_assert( CUDA_request == pPacket->flags );
	g_assert( CUDA_request == packet.flags );
}

int
main(int argc, char *argv[]){
	// initialize test program

	g_test_init(&argc, &argv, NULL);
	// hook up the test functions
	g_test_add_func("/test/testLibci/test_l_setMetThrReq", test_l_setMetThrReq);

	return g_test_run();
}

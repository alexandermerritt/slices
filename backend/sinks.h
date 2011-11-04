/**
 * @file sinks.h
 * @date 2011-10-27
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @brief TODO
 */
#ifndef _SINKS_H
#define _SINKS_H

#include <assembly.h>
#include <common/libregistration.h>

void localsink(asmid_t asmid, regid_t regid);
int nv_exec_pkt(volatile struct cuda_packet *pkt);

// TODO remote sink? something

#endif

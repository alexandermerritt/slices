/**
 * @file x86_system.h
 * @author Linux kernel developers
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-10-28
 * @brief File stolen from Linux kernel. Needed for memory barrier operations.
 */

#ifndef _ASM_X86_SYSTEM_H
#define _ASM_X86_SYSTEM_H

/* NOTE: these macros assume 64-bit processors.
 * Force strict CPU ordering.
 * And yes, this is required on UP too when we're talking
 * to devices.
 */
#define mb() 	asm volatile("mfence":::"memory")
#define rmb()	asm volatile("lfence":::"memory")
#define wmb()	asm volatile("sfence" ::: "memory")

#endif /* _ASM_X86_SYSTEM_H */

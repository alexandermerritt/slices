/**
 * @file shmgrp.c
 * @author Alex Merritt, merritt.alex@gatech.edu
 * @date 2011-11-05
 * @brief This file manages state associated with members of groups.
 */

#include <shmgrp.h>
#include <stdio.h>

/*-------------------------------------- INTERNAL TYPES ----------------------*/

//! State associated between a member and a group.
struct membership
{
	pid_t member_pid;
};

//! State associated with a group of members.
struct group
{
};

/*-------------------------------------- PUBLIC INTERFACE --------------------*/

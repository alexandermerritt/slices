#!/bin/bash

# Author: Alex Merritt, merritt.alex@gatech.edu
# This file is meant to clean system resources and files from an unclean
# termination of the assembly runtime and associated sink processes.

ASSEMBLY_SHMGRP_KEY=cudarpc

set -e
set -u

# Double-check with the user, just in case.
echo ""
echo "This script will purge resources and lingering processes"
echo "allocated by the assembly runtime which may not have been"
echo "removed due to an unclean termination."
echo ""
echo -n "Are you sure? [y/N] "

read confirm

if [[ ! "$confirm" == "y" ]]
then
	echo "Not confirmed."
	exit 0
fi

# Kill all runtime processes. There may be more than one lingering around.
# SIGINT is the signal these processes explicitly listen for.
RUNTIME_PIDS=`ps aux | grep runtime | grep -v grep | awk '{print $2}' | tr "\n" " "`
for RUNTIME_PID in $RUNTIME_PIDS
do
	echo Killing runtime $RUNTIME_PID
	kill -s SIGINT $RUNTIME_PID
	sleep 1 # Give the process time to cleanup.
done

# TODO Verify these processes no longer exist. If they do, send SIGKILL

# Kill all sink processes.
# SIGTERM is the signal these processes explicitly listen for.
SINK_PIDS=`ps aux | grep sink | grep -v grep | awk '{print $2}' | tr "\n" " "`
for SINK_PID in $SINK_PIDS
do
	echo Killing sink $SINK_PID
	kill -s SIGTERM $SINK_PID
	sleep 1 # Give the process time to cleanup.
done

# TODO Verify these processes no longer exist. If they do, send SIGKILL

# Remove registration and shm files.
rm -rf -v /tmp/shmgrp/${ASSEMBLY_SHMGRP_KEY}/
rm -f -v /dev/shm/shmgrp-${ASSEMBLY_SHMGRP_KEY}-*
rm -f -v /dev/shm/assm-export-*

echo ""
echo "Done."

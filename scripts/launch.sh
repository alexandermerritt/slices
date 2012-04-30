#! /usr/bin/env bash

set -e

# requires command-line argument(s) of course
exe="$HOME/code/shadowfax.git/exec.sh"

if [ -z $PBS_NODEFILE ]
then
	echo "Error: PBS_NODFILE undefined or file doesn't exist."
	exit 1
fi

mainNode=$(head -n 1 $PBS_NODEFILE)
echo "Launching main on $mainNode"
ssh -t $mainNode $exe main
sleep 1

for node in $(cat $PBS_NODEFILE | tr "\n" " ")
do
	if [ $node == $mainNode ]
	then
		continue
	fi
	echo "Launching minion on $node"
	ssh -t $node $exe minion $mainNode
	sleep 1
done

#! /usr/bin/env bash

hostname=$(hostname)
deathCmd="killall -s SIGINT runtime"
purgeCmd="~/code/shadowfax.git/backend/purge-system-resources.sh"

if [ -z $PBS_NODEFILE ]
then
	echo "Error: PBS_NODFILE undefined or file doesn't exist."
	exit 1
fi

mainNode=$(head -n 1 $PBS_NODEFILE)

for node in $(cat $PBS_NODEFILE | tr "\n" " ")
do
	if [ $node == $mainNode ]
	then
		continue
	fi
	echo "Halting minion on $node"
	ssh $node $deathCmd
	sleep 1
done

echo "Halting main on $mainNode"
ssh $mainNode $deathCmd

for node in $(cat $PBS_NODEFILE | tr "\n" " ")
do
	echo "Purging on $node"
	ssh $node $purgeCmd
	sleep 1
done

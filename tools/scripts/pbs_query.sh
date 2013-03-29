#! /usr/bin/env bash
[ -z "$PBS_ENVIRONMENT" ] &&
        echo "Error: PBS environment not defined." && exit 1

for node in $(sort -u $PBS_NODEFILE)
do
    pids=$(ssh -oStrictHostKeyChecking=no $node "pgrep -l shadowfax")
    [ -z "$pids" ] && continue
    echo $node $pids
done

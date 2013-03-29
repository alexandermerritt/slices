#! /usr/bin/env bash
[ -z "$PBS_ENVIRONMENT" ] &&
        echo "Error: PBS environment not defined." && exit 1

for node in $(sort -u $PBS_NODEFILE)
do
    pids=$(ssh -oStrictHostKeyChecking=no $node "pgrep shadowfax")
    [ -z "$pids" ] && continue
    echo "SIGINT --> $node $pids .."
    pids=$(ssh -oStrictHostKeyChecking=no $node "kill -s SIGINT $pids")
done

sleep 4

for node in $(sort -u $PBS_NODEFILE)
do
    pids=$(ssh -oStrictHostKeyChecking=no $node "pgrep shadowfax")
    [ -z "$pids" ] && continue
    echo "SIGKILL --> $node $pids .."
    pids=$(ssh -oStrictHostKeyChecking=no $node "kill -s SIGKILL $pids")
done

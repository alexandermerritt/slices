#! /usr/bin/env python

# Functions to help control Shadowfax runtime.

import os
import time
import time
import socket
import subprocess as sp

HOME = os.environ['HOME']
SHADOWFAX_CODE_DIR = HOME + '/code/shadowfax.git'
HOSTNAME = os.environ['HOSTNAME']

def build(dns):
	install = [HOME + '/bin/scons', '-Q', '-C', SHADOWFAX_CODE_DIR, 'network=sdp']
	ssh = ['ssh', dns]
	sp.check_call(ssh + install)

def launch(type='main', dns=HOSTNAME):
	if type not in ['main', 'minion']:
		raise ValueError('Invalid type')
	bin = SHADOWFAX_CODE_DIR + '/exec.sh'
	print('NOTE -- binding all Shadowfax binaries to NUMA domain 1')
	cmd = ['/usr/bin/numactl', '--cpunodebind=1', bin, type]
	if type == 'minion':
		cmd = cmd + dns
	cmd = ['ssh', dns] + cmd
	# this needs to have a shell invoked as the command executes a shell script
	sp.check_call(' '.join(cmd), shell=True)

def halt(dns):
	kill = ['killall', '-s', 'SIGINT', 'runtime']
	ssh = ['ssh', dns]
	sp.check_call(ssh + kill)

# first one assumed to be master
def start(nodeList):
	#build(nodeList[0]) # assumes all nodes share same directory via NFS
	launch('main', nodeList[0])
	time.sleep(1)
	for node in nodeList[1:]:
		launch('minion', node)
		time.sleep(1)

def stop(nodeList):
	for node in nodeList[1:]:
		halt(node)
		time.sleep(1)
	halt(nodeList[0])

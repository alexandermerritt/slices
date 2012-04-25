#! /usr/bin/env python

# Functions to help control Shadowfax runtime. You will need to edit this for
# the correct location of the source code and the network type to build with.
# To be used, this script should be placed in a directory that the environment
# variable PYTHONPATH points to. Or you can import it directly using a quoted
# relative path.

import os
import time
import socket
import subprocess as sp

HOME = os.environ['HOME']
SHADOWFAX_CODE_DIR = HOME + '/code/shadowfax.git'
HOSTNAME = os.environ['HOSTNAME']

def build(dns):
	install = [HOME + '/bin/scons', '-Q', '-C', SHADOWFAX_CODE_DIR, 'network=eth']
	ssh = ['ssh', dns]
	sp.call(ssh + install)

def launch(type='main', dns=HOSTNAME):
	if type not in ['main', 'minion']:
		raise ValueError('Invalid type')
	bin = SHADOWFAX_CODE_DIR + '/scripts/exec.sh'
	cmd = [bin, type]
	if type == 'minion':
		cmd = cmd + [HOSTNAME]
	cmd = ['ssh', dns] + cmd
	sp.call(' '.join(cmd), shell=True)

def halt(dns):
	kill = ['killall', '-s', 'SIGINT', 'runtime']
	ssh = ['ssh', dns]
	sp.call(ssh + kill)

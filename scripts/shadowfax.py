#! /usr/bin/env python

# Functions to help control Shadowfax runtime.

import os
import time
import socket
import subprocess as sp

HOME = os.environ['HOME']
SHADOWFAX_CODE_DIR = HOME + '/code/shadowfax.git'
HOSTNAME = os.environ['HOSTNAME']

def build(dns):
	install = [HOME + '/bin/scons', '-Q', '-C', SHADOWFAX_CODE_DIR, 'network=sdp']
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

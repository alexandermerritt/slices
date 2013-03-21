#! /usr/bin/env python

""" Build script for the Shadowfax code base.

	Modify the local_* variables to include header and library directory paths for
	locally installed software. Currently, ~/local/lib and ~/local/include are
	appended for Keeneland nodes.

	New machines may be added by extending the CUDA location configuration and
	the global ENV dict variable.
"""

import os
import commands
import sys

__author__ = "Alex Merritt"
__email__ = "merritt.alex@gatech.edu"

#
# Global variables
#
NODE_NAME = commands.getoutput('uname -n').split('.')[0]
ENV = {}
INSTALL_DIR = os.getcwd() + '/build'
#INSTALL_DIR = os.environ['HOME'] + '/exp/' # keeneland

#
# Configure CUDA location
#
cuda_root=''
if NODE_NAME.startswith('kid'):
	cuda_root = '/sw/keeneland/cuda/.3.2/linux_binary'
	NODE_NAME = 'kid'
elif NODE_NAME == 'prost':
	cuda_root = '/opt/cuda/'
elif NODE_NAME == 'shiva':
	cuda_root = '/usr/local/cuda'
elif NODE_NAME == 'ifrit':
	cuda_root = '/usr/local/cuda'
elif NODE_NAME == 'gunn':
	cuda_root = '/usr/local/cuda'
else:
	print("Build not configured for this node.")
	sys.exit(-1)

#
# Extract arguments
#
args = {}
args['debug'] = ARGUMENTS.get('debug', 0)
# Perform latency measurements of the code.
args['timing'] = ARGUMENTS.get('timing', 0)
# Timing, but without use of the backend; native passthrough only.
# XXX DO NOT run multi-threaded codes with timing_native
args['timing_native'] = ARGUMENTS.get('timing_native', 0)
args['network'] = ARGUMENTS.get('network', 'eth')
# Options you'd want to turn off
args['pipelining'] = ARGUMENTS.get('pipelining', 1)

#
# Configure environment
#
ccflags = ['-Wall', '-Wextra', '-Werror']
ccflags.append('-Winline')
ccflags.extend(['-Wno-unused-parameter', '-Wno-unused-function'])

if int(args['debug']):
	ccflags.append('-ggdb')
	ccflags.append('-DDEBUG')
else:
	ccflags.append('-O3')

if int(args['timing']):
	ccflags.append('-DTIMING')

if int(args['timing_native']):
	if not int(args['timing']):
		print('--> timing_native only valid with timing')
		sys.exit(1)
	ccflags.append('-DTIMING_NATIVE')

if args['network'] == 'eth':
	ccflags.append('-DNIC_ETHERNET')
elif args['network'] == 'sdp':
	ccflags.append('-DNIC_SDP')
else:
	print('--> network flag invalid')
	sys.exit(1)

if not int(args['pipelining']):
    ccflags.append('-DNO_PIPELINING')

# for anything you install locally, add/modify them here
home = os.environ['HOME']
local_lpath = [home + '/local/lib']
local_cpath = [home + '/local/include']
icc = '/opt/intel/composer_xe_2011_sp1.8.273/bin/intel64/icc'

# env values common across all files in project
libpath = [cuda_root + '/lib64', '/lib64']
cpath = [cuda_root + '/include', os.getcwd() + '/include']
libs = ['rt', 'dl']

# machine-specific paths
ENV['kid'] = Environment(CCFLAGS = ccflags, LIBS = libs)
ENV['kid'].Append(CPPPATH = cpath + local_cpath)
ENV['kid'].Append(LIBPATH = libpath + local_lpath)

ENV['prost'] = ENV['kid']

ENV['shiva'] = Environment(CC = 'gcc', CCFLAGS = ccflags, LIBS = libs)
ENV['shiva'].Append(CPPPATH = cpath)
ENV['shiva'].Append(LIBPATH = libpath)

ENV['ifrit'] = ENV['shiva']
ENV['gunn'] = ENV['shiva']

#
# Execute the build
#
Export('NODE_NAME', 'ENV', 'INSTALL_DIR')
SConscript(['interposer/SConstruct'])
if 0 == int(args['timing_native']):
    SConscript(['backend/SConstruct'])

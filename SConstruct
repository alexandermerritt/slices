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

#
# Configure CUDA location
#
cuda_root=''
if NODE_NAME.startswith('kid'):
	cuda_root = '/sw/keeneland/cuda/3.2/linux_binary'
	NODE_NAME = 'kid'
elif NODE_NAME == 'prost':
	cuda_root = '/opt/cuda/'
elif NODE_NAME == 'shiva':
	cuda_root = '/usr/local/cuda'
elif NODE_NAME == 'ifrit':
	cuda_root = '/usr/local/cuda'
else:
	print("Build not configured for this node.")
	sys.exit(-1)

#
# Extract arguments
#
args = {}
args['debug'] = ARGUMENTS.get('dbg', 0)
args['timing'] = ARGUMENTS.get('timing',0)

#
# Configure environment
#
ccflags = ['-Wall', '-Wextra', '-Werror']
ccflags.extend(['-Wno-unused-parameter', '-Wno-unused-function'])

if int(args['debug']) == 1:
	ccflags.append('-ggdb')
	ccflags.append('-DDEBUG')
else:
	ccflags.append('-O3')

if int(args['timing']) == 1:
	ccflags.append('-DTIMING')

# for anything you install locally, add/modify them here
home = os.environ['HOME']
local_lpath = [home + '/local/lib']
local_cpath = [home + '/local/include']

# env values common across all files in project
libpath = [cuda_root + '/lib64', '/lib64']
cpath = [cuda_root + '/include', os.getcwd() + '/include']
libs = ['rt', 'dl', 'shmgrp']

# machine-specific paths
ENV['kid'] = Environment(CCFLAGS = ccflags, LIBS = libs)
ENV['kid'].Append(CPPPATH = cpath + local_cpath)
ENV['kid'].Append(LIBPATH = libpath + local_lpath)

ENV['prost'] = ENV['kid']

ENV['shiva'] = Environment(CC = 'gcc4.4.4', CCFLAGS = ccflags, LIBS = libs)
ENV['shiva'].Append(CPPPATH = cpath)
ENV['shiva'].Append(LIBPATH = libpath)

ENV['ifrit'] = ENV['shiva']

#
# Execute the build
#
Export('NODE_NAME', 'ENV')
SConscript(['interposer/SConstruct', 'backend/SConstruct'])

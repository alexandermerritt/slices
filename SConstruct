#! /usr/bin/env python

import os
import commands
import sys

__author__ = "Alex Merritt"
__email__ = "merritt.alex@gatech.edu"

cuda_root = os.environ.get('CUDA_ROOT', '/usr/local/cuda')

arg_debug       = int(ARGUMENTS.get('debug', 0))
arg_network     = ARGUMENTS.get('network', 'eth')
arg_pipelining = int(ARGUMENTS.get('pipelining', 1))

ccflags = ['-Wall', '-Wextra', '-Werror']
ccflags.append('-Winline')
ccflags.extend(['-Wno-unused-parameter', '-Wno-unused-function'])

home = os.environ['HOME']
local_lpath = [home + '/local/lib']
local_cpath = [home + '/local/include']

libpath = [cuda_root + '/lib64', '/lib64']
cpath = [cuda_root + '/include', os.getcwd() + '/include']
libs = ['rt', 'dl']

if arg_debug:
    ccflags.append('-ggdb')
    ccflags.append('-DDEBUG')
    ccflags.append('-O0')
    #libs.append('mcheck')
else:
    ccflags.append('-O3')

if arg_network == 'eth':
    ccflags.append('-DNIC_ETHERNET')
elif arg_network == 'sdp':
    ccflags.append('-DNIC_SDP')
else:
    print('--> network flag invalid')
    sys.exit(1)

if not arg_pipelining:
    ccflags.append('-DNO_PIPELINING')

env = Environment(CC = 'clang', CCFLAGS = ccflags, LIBS = libs)
env.Append(CPPPATH = cpath)
env.Append(LIBPATH = libpath)

Export('env')
SConscript(['src/SConstruct'])

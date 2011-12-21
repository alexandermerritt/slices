#! /usr/bin/env python

"""@file SConstruct
   @author Magdalena Slawinska magg@gatech.edu
   @author Alex Merritt, merritt.alex@gatech.edu
   @date 2011-02-15
"""
import os
import sys
import getpass
import socket

Help("""
   $ scons -Q   # builds quietly the program
   $ scons -c   # cleans the program

   # run with the debug compilation options enabled
   $ scons --set-debug-flags  
   
   # to run with the optimization compilation flag enabled
   $ scons --set-opt-flags  
   
   For defined configurations see function
          
   set_build_variables()
    """)

# default values
GCC = { 
        'CC' : 'gcc',
        'CCFLAGS' :  ['-fPIC', '-Wall', '-Wextra', '-Wno-unused-parameter', '-std=gnu99'],
        'CCDEBUGFLAGS' : ['-ggdb', '-g', '-DDEBUG'],
        'CCOPTFLAGS' : ['-O3']
      }

CUDA_ROOT = None

# where is the installation directory
INSTALL_DIR = os.getcwd() + '/distro'

#################################################
# helper functions
#################################################
def set_build_context():
    """
    tries to get a configuration characteristic string
    currently: three first letters of the machine hostname
    concatenated with '_' and concatenated with the user name
    e.g. kidlogin.nics.utk.edu and a user john
    kid_john
    """
    
    nodename = socket.gethostname()
    username = getpass.getuser()

    return nodename[:3] + '_' + username
    

#################################################
# helper functions
#################################################
def set_build_variables(build_ctx):
    """
      sets the build variables automatically depending on the system you
      are working on
      @param build_ctx: the detected build context (string) 
    """
    global CUDA_ROOT
    global GCC
    
    # configuration for kid_smagg
    if build_ctx == 'kid_smagg':
        print(build_ctx + ' build context detected ...')
        CUDA_ROOT = '/sw/keeneland/cuda/3.2/linux_binary'
        
        # configuration for keeneland
    elif build_ctx == 'kid_merritt' :
        print(build_ctx + ' build context detected ...')
        CUDA_ROOT = '/sw/keeneland/cuda/3.2/linux_binary'
        
    # Custom machine at Georgia Tech
    elif build_ctx == 'shi_alex' or build_ctx == 'ifr_alex':
        print(build_ctx + ' detected ...')
        CUDA_ROOT = '/usr/local/cuda'
        GCC['CC'] = 'gcc4.4.4'
    
    else:
        print('''ERROR: no build configuration detected. Define
        a build configuration for this machine''')
        sys.exit(-1)

def determine_extra_compiler_options(env):
    """
    determines the additional compiler options such as 
    debug or optimization and appends it to the base environment
    @param env: Environment - the environment to which the new things will be appended 
    """
    debug = GetOption('DEBUG')
    optimized = GetOption('OPTIMIZED')
    ok = True
    if None == debug:
        debug = False
    if None == optimized:
        optimized == False

    if debug and optimized:
        print('Contradictory build options: Debug and Optimized flags set. ')
        sys.exit(0)
    
    if False == debug and False == optimized:
        print('Neither optimization nor debug compilation options enabled')
    elif debug: 
        env.Append(CCFLAGS = GCC['CCDEBUGFLAGS'])
        print('Debug compilation option enabled')
    elif optimized:
        env.Append(CCFLAGS = GCC['CCOPTFLAGS'])
        print('Optimized compilation enabled')
    
    
def print_vars():
    """
       print variables that will be used for the build
    """
    global CUDA_ROOT
    global INSTALL_DIR
    global GCC
    
    print('IMPORTANT BUILD VARIABLES')
    print('=========================')
    
    print('CUDA_ROOT= ' + CUDA_ROOT)
    print('INSTALL_DIR=' + INSTALL_DIR)

    print('GCC base settings:')
    for key in GCC.keys():
        sys.stdout.write('\t' + key + '=')
        print(GCC[key])
    sys.stdout.write('Debug options appended: ') 
    print( GetOption('DEBUG'))    
    sys.stdout.write('Optimized options appended: ')
    print(       GetOption('OPTIMIZED'))
    print('=========================')
    

###################
# the actual build

AddOption('--set-debug-flags', action='store_true', dest='DEBUG',  
          help='if present the debug compilation options will be enabled')
AddOption('--set-opt-flags', action='store_true', dest='OPTIMIZED', 
          help='if present, the optimized compilation options will be enabled')


# determine the build configuration    
build_ctx = set_build_context()
print('Detected build context: ' + build_ctx)

set_build_variables(build_ctx)

# the environment for building the ib_rdma
BASE_ENV = Environment(
    CPPPATH = ['../include'],
    CCFLAGS = GCC['CCFLAGS'],
    CC = GCC['CC']
    )          

determine_extra_compiler_options(BASE_ENV)

print_vars()


# export variables to other scripts
Export('CUDA_ROOT', 
       'GCC',
       'INSTALL_DIR',        
       'BASE_ENV')
               
# call all scripts
SConscript('shmgrp/SConstruct')
SConscript('shmgrp/test/SConstruct')
SConscript('interposer/SConstruct')
SConscript('backend/SConstruct')

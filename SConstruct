#! /usr/bin/env python

"""@file SConstruct
   @author Magdalena Slawinska magg@gatech.edu
   @author Alex Merritt, merritt.alex@gatech.edu
   @date 2011-02-15
   @brief Building kidron-utils-remote-cuda-exec-2
"""
import os
import commands

Help("""
   Type: 'scons -Q' to build the production program,
         'scons -c' to clean the program
    """)

# Customize the build environment based on the machine invoking the script.
nodename = commands.getoutput('uname -n')
if nodename.startswith('shiva') :
    # Custom machine at Georgia Tech
    CUDA_ROOT = '/usr/local/cuda/'
    CUNIT212 = '/usr/local/lib/'
    GLIB20 = '/usr/include/glib-2.0/'
elif nodename.startswith('ifrit') :
    # Custom machine at Georgia Tech, same as shiva
    CUDA_ROOT = '/usr/local/cuda/'
    CUNIT212 = '/usr/local/lib/'
    GLIB20 = '/usr/include/glib-2.0/'
elif nodename.startswith('prost') :
    # Dell machine at Georgia Tech
    CUDA_ROOT = '/opt/cuda/'
    CUNIT212 = '/opt/cunit212/'
    GLIB20   = '/opt/glib-2.28.7/'
else :
    # Assume compilation on a Keeneland node.
    # TODO: you should support also GLIB variable
    CUDA_ROOT = '/sw/keeneland/cuda/3.2/linux_binary/'
    CUNIT212 = '/nics/d/home/smagg/opt/cunit212/'
    GLIB20='/nics/d/home/smagg/opt/glib-2.28.7/'

if not os.path.exists(CUDA_ROOT):
        print """CUDA_ROOT=""", CUDA_ROOT, """ does not exist!"""

if not os.path.exists(CUNIT212):
        print """CUINT212=""", CUNIT212, """ does not exist!"""

if not os.path.exists(GLIB20):
        print """GLIB20=""", GLIB20, """ does not exist!"""

# export variables to other scripts
Export( 'CUDA_ROOT', 'CUNIT212', 'GLIB20' )

# call all scripts
SConscript([
        'cuda-app/SConstruct',		# it doesn't depend on anything
        'interposer/SConstruct',    # it compiles a bunch of stuff
        'backend/SConstruct'			
        ])


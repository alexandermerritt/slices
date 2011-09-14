"""@file SConstruct
   @author Magdalena Slawinska magg@gatech.edu
   @date 2011-02-15
   @brief Building kidron-utils-remote-cuda-exec-2
"""
import os

Help("""
   Type: 'scons -Q' to build the production program,
         'scons -c' to clean the program
    """)

# keeneland
#CUDA_ROOT = '/sw/keeneland/cuda/3.2/linux_binary/'
#INIPARSER = '/nics/d/home/smagg/src/iniparser/'
#CUNIT212 = '/nics/d/home/smagg/opt/cunit212/'
# TODO: you should support also GLIB variable
# prost georgia tech
CUDA_ROOT = '/opt/cuda/'
INIPARSER = '/home/magg/src/iniparser/'
CUNIT212 = '/opt/cunit212/'
GLIB20   = '/opt/glib-2.28.7/'

if not os.path.exists(INIPARSER):
        print INIPARSER, """does not exist!"""
                        
if not os.path.exists(CUDA_ROOT):
        print CUDA_ROOT, """does not exist!"""


if not os.path.exists(CUNIT212):
        print CUNIT212, """does not exist!"""

if not os.path.exists(GLIB20):
        print GLIB20, """ does not exist! """
# export variables to other scripts
Export( 'CUDA_ROOT', 'INIPARSER', 'CUNIT212', 'GLIB20' )

					  
# call all scripts
SConscript([
'cuda-app/SConstruct',		# it doesn't depend on anything
'interposer/SConstruct',    # it compiles a bunch of stuff
'backend/SConstruct'			
])
                  

 
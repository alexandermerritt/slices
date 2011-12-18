#! /usr/bin/env python

"""@file SConstruct
   @author Magdalena Slawinska magg@gatech.edu
   @author Alex Merritt, merritt.alex@gatech.edu
   @date 2011-02-15
   @brief Building kidron-utils-remote-cuda-exec-2
"""
import os
import commands
import sys

Help("""
   Type: 'scons -Q' to build the production program,
         'scons -c' to clean the program
   Currently automatically detected environments: 
   - keeneland (NICS, UTK) 
   - prost     (Georgia Tech)
   - shiva     (Georgia TEch)
   - ifrit     (Georgia TEch)
   If you want to add another environment check the function
   build_variables_set(). Currently it is based on the
   hostnames you are building on.
    """)


""" 
  required variables for this building system
  if you need to add another configuration you need to modify the 
  setBuildVariables()
"""

# points to the directory where the CUDA root is
CUDA_ROOT=None

def get_platform_characteristic_str():
    """
       intended to get the characteristic string to allow for automatic
       recognition of the platform and applied customized build environment
       @return: the characteristic string for the platform
       @rtype: string
    """
    nodename = commands.getoutput('uname -n')
    return nodename
    

#################################################
# helper functions
#################################################
def build_variables_set():
    """
      sets the build variables automatically depending on the system you
      are working on
    """
    global CUDA_ROOT
        
    nodename = get_platform_characteristic_str()
    print('The configuration will be applied for: ' + nodename)
    
    # configuration for keeneland
    if ( nodename.startswith('kid') ):
        print('kid prefix detected ...')
        CUDA_ROOT = '/sw/keeneland/cuda/3.2/linux_binary/'
    
    # custom machine at Georgia Tech configuration for prost
    if ( nodename.startswith('prost')):
        print('prost prefix detected ...')
        CUDA_ROOT = '/opt/cuda/'
    
    # Custom machine at Georgia Tech
    if nodename.startswith('shiva'):
 	print('shiva prefix detected ...')
        CUDA_ROOT = '/usr/local/cuda/'
    
    # Custom machine at Georgia Tech, same as shiva
    if nodename.startswith('ifrit') :
	print('ifrit prefix detected ...')
        CUDA_ROOT = '/usr/local/cuda/'


def variable_check_exit(var_name, var):
    """
        checks if the variable is correctly set and quits the script if not
        @param var_name: The name of the variable to be checked 
        @param var: The variable that supposed to be a path to the directory 
    """
    if var == None :
        print(var_name +  ' not set. You have to set it in build_variables_set() in this script')
        sys.exit(-1)
    if not os.path.exists(var):
        print(var_name + '= ' + var + ' does not exist!')
        sys.exit(-1)
    print(var_name + '= ' +  var)


def build_variables_print():
    """
      prints the build variables or exits the script if they are not set
    """
    variable_check_exit('CUDA_ROOT', CUDA_ROOT)

#######################################
# start actual execution script
#######################################

# set build variables    
build_variables_set()
# check if the variables are set and directories exist and print them
build_variables_print()


# export variables to other scripts
Export( 'CUDA_ROOT')

					  
# call all scripts
SConscript([
#        'cuda-app/SConstruct', # it doesn't depend on anything
        'interposer/SConstruct',    # it compiles a bunch of stuff
        'backend/SConstruct'			
        ])


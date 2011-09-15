"""
 @file devel_info.py
 @date: Aug 29, 2011
 @author: Magda Slawinska, aka Magic Magg, magg dot gatech at gmail dot edu

 This file allows to get info about the development environment for this system.
"""

import os

print('Info system developed by Magda Slawinska, aka Magic Magg, 2011')
print('The information about the development environment for the '
      'Kidron Utils dSimons')
print('************>  date')
os.system('date')

print('************>  hostname')
os.system('hostname')

print('************> SOFTWARE')
print('************>scons --version')
os.system('scons --version')
print( '************> gcc -v')
os.system('gcc -v')
print( '************> nvcc --version')
os.system('nvcc --version')

print('************> SYSTEM AND HARDWARE')
print '************> uname -a'
os.system('uname -a')
print '************> nvidia-smi -q | head'
#os.system('nvidia-settings --help')
os.system('nvidia-smi -q | head')
print '************> cat /etc/redhat-release'
os.system('cat /etc/redhat-release')
print '************>  cat /proc/cpuinfo | head'
os.system('cat /proc/cpuinfo | head')

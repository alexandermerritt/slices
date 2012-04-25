#! /usr/bin/env bash

set -e

CODE_DIR=$HOME/code/shadowfax.git

# Because when invoked via ssh none of this shit is available, so have to source
# it all again.
source /etc/profile
source /etc/bashrc
source ~/.bash_profile

cd $CODE_DIR/build/bin
./runtime $@


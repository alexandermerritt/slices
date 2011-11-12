## @file Makefile
## @author Alex Merritt, merritt.alex@gatech.edu
## @date 2011-11-08
## @brief Makefile to build the shmgrp dynamic library

CC=gcc
CCFLAGS=-fPIC -O0 -ggdb -Wall -Werror -Wextra -Wno-unused-parameter -Iinclude/
LDFLAGS=-lrt -shared -Wl,-soname,libshmgrp.so

all:	lib/libshmgrp.so

lib/libshmgrp.so:	src/common.o src/leader.o src/member.o
	mkdir -p lib
	$(CC) $(LDFLAGS) -o $@ $^
	echo Append lib/ to LD_LIBRARY_PATH before executing your program.

src/common.o:	src/common.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/leader.o:	src/leader.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/member.o:	src/member.c
	$(CC) $(CCFLAGS) -c -o $@ $<

clean:
	rm -rf lib/ src/*.o

## @file Makefile
## @author Alex Merritt, merritt.alex@gatech.edu
## @date 2011-11-08
## @brief Makefile to build the shmgrp dynamic library

PREFIX=/usr/local
SONAME=libshmgrp.so

CC=gcc
CCFLAGS=-fPIC -ggdb -Wall -Werror -Wextra -Wno-unused-parameter -Iinclude/
LDFLAGS=-lrt -shared -Wl,-soname,$(SONAME)

all:	lib/$(SONAME) test

.PHONY: all

lib/$(SONAME):	src/common.o src/leader.o src/member.o
	mkdir -p lib
	$(CC) $(LDFLAGS) -o $@ $^

src/common.o:	src/common.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/leader.o:	src/leader.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/member.o:	src/member.c
	$(CC) $(CCFLAGS) -c -o $@ $<

test:
	make -C test/

.PHONY:	test

clean:
	rm -rf lib/ src/*.o
	make -C test/ clean

.PHONY: clean

install: all
	install -m 0444 include/shmgrp.h $(PREFIX)/include
	install -m 0555 lib/$(SONAME) $(PREFIX)/lib
	ldconfig

.PHONY: install

uninstall:
	rm -f $(PREFIX)/include/shmgrp.h
	rm -f $(PREFIX)/lib/$(SONAME)
	ldconfig

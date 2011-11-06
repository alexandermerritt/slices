CC=gcc
CCFLAGS=-fPIC -O2 -Wall -Werror -Wextra -Wno-unused-parameter -Iinclude/
LDFLAGS=-lrt -shared -Wl,-soname,libshmgrp.so

all:	lib/libshmgrp.so

lib/libshmgrp.so:	src/common.o src/leader.o src/member.o
	mkdir -p lib
	$(CC) $(LDFLAGS) -o $@ $^

src/common.o:	src/common.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/leader.o:	src/leader.c
	$(CC) $(CCFLAGS) -c -o $@ $<

src/member.o:	src/member.c
	$(CC) $(CCFLAGS) -c -o $@ $<

clean:
	rm -rf lib/ src/*.o

CC=gcc
#CCFLAGS=-I../include -ggdb -O0
CCFLAGS=-I../include -O2
LDFLAGS=-L../lib -lshmgrp

all:	test_leader test_member

test_leader:	test_leader.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

test_member:	test_member.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o test_leader test_member

# NOTE: Execute the test programs like so:
#	LD_LIBRARY_PATH=../lib ./test_leader

CC=gcc
CCFLAGS=-I../include -ggdb -O0
#CCFLAGS=-I../include -O2
LDFLAGS=-L../lib -lshmgrp

BINARIES=test_leader test_member hacktest_leader hacktest_child

all:	$(BINARIES)

test_leader:	test_leader.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

test_member:	test_member.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

hacktest_leader:	hacktest_leader.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

hacktest_child:	hacktest_child.c
	$(CC) $(CCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.o $(BINARIES)

# NOTE: Execute the test programs like so:
#	LD_LIBRARY_PATH=../lib ./test_leader

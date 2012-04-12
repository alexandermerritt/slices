CC=g++
CCFLAGS=-O3 -std=c++0x -fpermissive # fpermissive to allow fill_ops void* warnings
INC=-I/usr/local/cuda/include -I../../include

all:	libcudainst.so
.PHONY:	all

libcudainst.so:	runtime_api.o trace.o
	$(CC) -shared -Wl,-soname,$@ -o $@ $^

runtime_api.o:	runtime_api.cpp
	$(CC) $(CCFLAGS) -fpic -c $(INC) $<

trace.o:	trace.cpp
	$(CC) $(CCFLAGS) -fpic -c $(INC) $<

clean:
	rm -f *.o *.so
.PHONY:	clean

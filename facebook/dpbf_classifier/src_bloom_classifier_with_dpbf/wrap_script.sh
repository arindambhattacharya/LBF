#!/bin/bash

swig -python dpbf.i
g++ -std=c++14 -c -fpic dpbf_wrap.c dpbf.c -I/usr/include/python3.6m
g++ -std=c++14 -shared dpbf.o dpbf_wrap.o -o _dpbf.so
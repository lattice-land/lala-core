#!/bin/sh

mkdir -p build/cpu-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=OFF -Bbuild/cpu-debug &&
cmake --build build/cpu-debug &&
(cd build/cpu-debug;
ctest;
cd ../..)

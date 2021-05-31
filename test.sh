#!/bin/sh

mkdir -p build/cpu-release
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=OFF -Bbuild/cpu-release &&
cmake --build build/cpu-release &&
(cd build/cpu-release;
ctest;
cd ../..)

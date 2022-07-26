#!/bin/bash

git submodule init
git submodule update
cd vcpkg
./bootstrap-vcpkg.sh


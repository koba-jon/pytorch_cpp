#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
LIST="${SCRIPT_DIR}/build_list.txt"
THREADS=7

# Create Makefile
while read dir
do
    cd ${SCRIPT_DIR}/../${dir}
    if [ -e build ]; then
        rm -r build
    fi
    mkdir build
    cd build
    cmake ..
done < ${LIST}

# Create Executable Files
while read dir
do
    cd ${SCRIPT_DIR}/../${dir}/build
    make -j${THREADS}
done < ${LIST}


cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(YOLOv3 CXX)

# Directory Name
set(SUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../utils)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Create Executable File
set(SRCS
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/train.cpp
    ${SRC_DIR}/valid.cpp
    ${SRC_DIR}/test.cpp
    ${SRC_DIR}/detect.cpp
    ${SRC_DIR}/demo.cpp
    ${SRC_DIR}/loss.cpp
    ${SRC_DIR}/networks.cpp
    ${SRC_DIR}/detector.cpp
    ${SRC_DIR}/augmentation.cpp
)

add_subdirectory(${SUB_DIR} build)

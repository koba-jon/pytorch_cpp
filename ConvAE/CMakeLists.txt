cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
set(PROJ_NAME ConvAE)

# LibTorch
set(LIBTORCH_DIR $ENV{HOME}/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# Directory Name
set(SRC_DIR ./src)
set(INCLUDE_DIR ./src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../)

# Find Package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

# For OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Set Include Directories
set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${INCLUDE_DIR}
)

# Set Libraries
set(LIBRARIES
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Set Source Code
add_executable(${PROJ_NAME}
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/train.cpp
    ${SRC_DIR}/valid.cpp
    ${SRC_DIR}/test.cpp
    ${SRC_DIR}/loss.cpp
    ${SRC_DIR}/networks.cpp
    ${SRC_DIR}/transforms.cpp
    ${SRC_DIR}/datasets.cpp
    ${SRC_DIR}/dataloader.cpp
    ${SRC_DIR}/visualizer.cpp
    ${SRC_DIR}/progress.cpp
)

# Link
include_directories(${INCLUDE_DIRS})
target_link_libraries(${PROJ_NAME} ${LIBRARIES})
set_property(TARGET ${PROJ_NAME} PROPERTY CXX_STANDARD 17)


# Display Message

message(STATUS "")
message(STATUS "~~~ PyTorch Information ~~~")
message(STATUS "${TORCH_INCLUDE_DIRS};")
message(STATUS "${TORCH_LIBRARIES};")
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~~")
message(STATUS "")

message(STATUS "~~~ OpenCV Information ~~~")
message(STATUS "${OpenCV_INCLUDE_DIRS};")
message(STATUS "${OpenCV_LIBRARIES};")
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~")
message(STATUS "")

message(STATUS "~~~ Boost Information ~~~")
message(STATUS "${Boost_INCLUDE_DIRS};")
message(STATUS "${Boost_LIBRARIES};")
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~")
message(STATUS "")

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# LibTorch
set(LIBTORCH_DIR $ENV{HOME}/libtorch)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})

# Set Compiler Options
set(CXX_LIBRARIES "stdc++fs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall")
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
    set(CMAKE_CXX_COMPILER "g++-8")
endif ()

# Directory Name
set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Find Package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)
find_package(PNG REQUIRED)

# For OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Set Include Directories
set(INCLUDE_DIRS
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
    ${UTILS_DIR}
    ${SRC_DIR}
)

# Set Libraries
set(LIBRARIES
    ${CXX_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${PNG_LIBRARIES}
)

# Set Utility Code
set(UTILITY
    ${UTILS_DIR}/transforms.cpp
    ${UTILS_DIR}/datasets.cpp
    ${UTILS_DIR}/dataloader.cpp
    ${UTILS_DIR}/losses.cpp
    ${UTILS_DIR}/visualizer.cpp
    ${UTILS_DIR}/progress.cpp
)

# Link
add_executable(${PROJECT_NAME} ${SRCS} ${UTILITY})
include_directories(${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)


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

message(STATUS "~~~ libpng Information ~~~")
message(STATUS "${PNG_INCLUDE_DIRS};")
message(STATUS "${PNG_LIBRARIES};")
message(STATUS "~~~~~~~~~~~~~~~~~~~~~~~~~~")
message(STATUS "")


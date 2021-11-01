cmake_minimum_required(VERSION 3.15)
project(clustering)
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
# set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)

find_package(OpenCV REQUIRED)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_executable(clustering ${DIR_SRCS} )
target_link_libraries(clustering PRIVATE
    ${OpenCV_LIBS}
)



# project
cmake_minimum_required (VERSION 2.6)
project (HomeworkVO)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV)
find_package(Eigen3 REQUIRED)

# include directory
include_directories(
  src/
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

aux_source_directory(src DIR_SRCS)

# make release version
set(CMAKE_BUILD_TYPE Release)

# demo program
add_executable(HomeworkVO ${DIR_SRCS})
target_link_libraries (HomeworkVO ${OpenCV_LIBS})


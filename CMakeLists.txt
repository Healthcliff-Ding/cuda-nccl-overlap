cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NcclOverlapTestMinimal LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GoogleTest Preparation - Code block copied from
# https://google.github.io/googletest/quickstart-cmake.html
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG release-1.12.1
# )
# FetchContent_MakeAvailable(googletest)

# fetch latest argparse
# FetchContent_Declare(
#   argparse
#   GIT_REPOSITORY https://github.com/p-ranav/argparse.git
# )
# FetchContent_MakeAvailable(argparse)

if(DEFINED ENV{CONDA_PREFIX})
  # use conda environment
  link_directories($ENV{CONDA_PREFIX}/lib)
  include_directories($ENV{CONDA_PREFIX}/include)
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

if(DEFINED ENV{CUDA_HOME})
  set(CUDAToolkit_ROOT $ENV{CUDA_HOME})
else()
  set(CUDAToolkit_ROOT /usr/local/cuda-12.1)
endif()

find_package(CUDAToolkit 11.4 REQUIRED)

# gcc >= 8 is required, we do not support other compilers
if((NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0))
  message(FATAL_ERROR "GCC 8.0 or higher is required")
endif()

# Add filesystem library for gcc < 9
link_libraries("$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>")

# Set up C++ standard
set(CXX_STD "20" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Switch between release mode and debug mode
# The user can use `-DBUILD_MODE=DEBUG` or `-DBUILD_MODE=RELEASE` to
# choose the build mode.
# If no option is provided, default to debug mode
if(BUILD_MODE)
  string(TOUPPER ${BUILD_MODE} BUILD_MODE)

  if(BUILD_MODE STREQUAL "DEBUG")
    set(DEBUG ON)
  elseif(BUILD_MODE STREQUAL "RELEASE")
    set(RELEASE ON)
  else()
    message(FATAL_ERROR "Unknown build mode: ${BUILD_MODE}")
  endif()
else()
  message("No build type selected, defaulting to RELEASE mode")
  message("Use -DBUILD_MODE=DEBUG or -DBUILD_MODE=RELEASE to specify build type")
  set(RELEASE ON)
endif()

# Set up C++ flag and CUDA flag
if(DEBUG)
  message("Building in debug mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DDEBUG")
else()
  message("Building in release mode")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --prec-div=false")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DRELEASE")
endif()

# Set up COMMON_HEADER_DIRS and COMMON_LIB_DIRS
set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
)
set(COMMON_LIB_DIRS "")

# Set up MPI and NCCL for multi-GPU communication
message("Building with MPI and NCCL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(MKL_MPI "openmpi")
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

# Add MPI and NCCL into COMMON_HEADER_DIRS & COMMON_LIB_DIRS
list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH} ${NCCL_INCLUDE_DIR})
list(APPEND COMMON_LIB_DIRS ${MPI_LIBRARIES} ${NCCL_LIBRARIES})

set(COMMON_LIBS CUDA::cudart)

# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})

# use CXX11_ABI
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

# Let all executable targets go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)

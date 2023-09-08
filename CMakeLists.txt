cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

project(lala-core
  HOMEPAGE_URL "https://github.com/lattice-land/lala-core"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" OFF)

# Cuda-battery dependency

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(cuda_battery SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../cuda-battery/")
else()
  FetchContent_Declare(
    cuda_battery
    GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
    GIT_TAG        v1.1.0
  )
endif()
FetchContent_MakeAvailable(cuda_battery)

# Preparing the library

add_library(lala_core INTERFACE)
target_include_directories(lala_core INTERFACE include)
target_link_libraries(lala_core INTERFACE cuda_battery)
target_compile_options(lala_core INTERFACE
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:-diag-suppress 20011,20014>" # useless warnings for us due to many template functions working on both CPU/GPU depending on the allocator (calling __host__ from __host__ __device__ is forbidden).
)

# Documentation

find_package(Doxygen REQUIRED doxygen)

set(DOXYGEN_PROJECT_NAME "Lattice Land Core Library")
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
set(DOXYGEN_CASE_SENSE_NAMES NO)
set(DOXYGEN_CLASS_DIAGRAMS NO)
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_FILE_PATTERNS *.hpp)
set(DOXYGEN_GENERATE_TREEVIEW YES)
set(DOXYGEN_HIDE_IN_BODY_DOCS YES)
set(DOXYGEN_QUIET YES)
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SORT_BY_SCOPE_NAME YES)
set(DOXYGEN_SORT_MEMBER_DOCS NO)
set(DOXYGEN_SOURCE_BROWSER NO)
set(DOXYGEN_STRIP_CODE_COMMENTS NO)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_EXTRA_PACKAGES stmaryrd)
set(DOXYGEN_EXCLUDE_SYMBOLS = impl)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_PREDEFINED __NVCC__)
set(DOXYGEN_REFERENCED_BY_RELATION NO)
set(DOXYGEN_REFERENCES_RELATION NO)

doxygen_add_docs(doc_lalacore
                 "include/"
                 ALL
                 COMMENT "Generate HTML documentation")

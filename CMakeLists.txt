cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Targeted CUDA architectures")
endif()

if(MSVC)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Available configuration types to select")
endif()

project(lala-core
  HOMEPAGE_URL "https://github.com/lattice-land/lala-core"
  LANGUAGES CUDA CXX)
option(GPU "GPU" ON)
option(LOCAL_DEPS "LOCAL_DEPS" OFF)
option(LALA_CORE_BUILD_TESTS "LALA_CORE_BUILD_TESTS" OFF)

# Cuda-battery dependency

include(FetchContent)

if(LOCAL_DEPS)
  FetchContent_Declare(cuda_battery SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../cuda-battery/")
  FetchContent_Declare(cccl SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../cccl/")
else()
  FetchContent_Declare(
    cuda_battery
    GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
    GIT_TAG        v1.4.0
  )
  FetchContent_Declare(
    cccl
    GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
    GIT_TAG        v2.3.1
    GIT_SHALLOW    1
  )

endif()
FetchContent_MakeAvailable(cuda_battery cccl)

# Preparing the library

if(NOT LALA_CORE_BUILD_TESTS) # For tests, lala_core will be built through lala_parsing (which depends on lala_core).
  if(NOT CUDA_PATH)
    set(CUDA_PATH "/usr/local/cuda")
  endif()
  add_library(lala_core INTERFACE)
  target_include_directories(lala_core INTERFACE "${CUDA_PATH}/include" include)
  target_link_libraries(lala_core INTERFACE cuda_battery CCCL::CCCL)
  target_compile_options(lala_core INTERFACE
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:SHELL:-diag-suppress 20011,20014>" # useless warnings for us due to many template functions working on both CPU/GPU depending on the allocator (calling __host__ from __host__ __device__ is forbidden).
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:SHELL:--compiler-options /utf-8>"
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>,$<CXX_COMPILER_ID:MSVC>>:SHELL:-diag-suppress 2219>" # Suppress warnings from nvcc about using UTF-8 "\uXXXX" strings in ast.hpp and primitive_upset.hpp
  )
endif()

if(LALA_CORE_BUILD_TESTS)

# Google Test dependency

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW 1
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

set(LALA_CORE_BUILD_TESTS OFF)
if(LOCAL_DEPS)
  FetchContent_Declare(lala_parsing SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../lala-parsing/")
else()
  FetchContent_Declare(
    lala_parsing
    GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
    GIT_TAG        v1.1.7
  )
endif()
FetchContent_MakeAvailable(lala_parsing)
set(LALA_CORE_BUILD_TESTS ON)

# I. CPU Tests (ending with "_test.cpp")
file(GLOB cpu_test_files tests/src/*_test.cpp)
foreach(file ${cpu_test_files})
  cmake_path(GET file STEM test_name)
  add_executable(${test_name} ${file})
  target_include_directories(${test_name} PRIVATE tests/include)
  target_link_libraries(${test_name} lala_parsing gtest_main)
  gtest_discover_tests(${test_name})
endforeach()

# II. GPU Tests (ending with "_gpu.cpp")
if(GPU)
  file(GLOB gpu_test_files tests/src/*_gpu.cpp)
  foreach(file ${gpu_test_files})
    cmake_path(GET file STEM test_name)
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
    add_executable(${test_name} ${file})
    target_include_directories(${test_name} PRIVATE tests/include)
    target_link_libraries(${test_name} lala_parsing)
    target_link_options(${test_name} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
  endforeach()
endif()

endif()

# Documentation

if(NOT LALA_CORE_BUILD_TESTS)

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
endif()

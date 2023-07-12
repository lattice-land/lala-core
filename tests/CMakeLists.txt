cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(lala-core-test
  VERSION 1.0.0
  HOMEPAGE_URL "https://github.com/lattice-land/lala-core"
  LANGUAGES CUDA CXX)

# Lala-parsing dependency.

include(FetchContent)

FetchContent_Declare(
  lala_parsing
  GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
  GIT_TAG        main
)
FetchContent_MakeAvailable(lala_parsing)

# Testing

# Google Test dependency

FetchContent_Declare(
  googletest
  GIT_REPOSITORY "https://github.com/google/googletest"
  GIT_TAG v1.13.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# I. CPU Tests (ending with "_test.cpp")
file(GLOB cpu_test_files src/*_test.cpp)
foreach(file ${cpu_test_files})
  cmake_path(GET file STEM test_name)
  add_executable(${test_name} ${file})
  target_include_directories(${test_name} PRIVATE include)
  target_link_libraries(${test_name} lala_parsing gtest_main)
  gtest_discover_tests(${test_name})
endforeach()

# II. GPU Tests (ending with "_gpu.cpp")
if(GPU)
  file(GLOB gpu_test_files src/*_gpu.cpp)
  foreach(file ${gpu_test_files})
    cmake_path(GET file STEM test_name)
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
    add_executable(${test_name} ${file})
    target_include_directories(${test_name} PRIVATE include)
    target_link_libraries(${test_name} lala_parsing)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
  endforeach()
endif()


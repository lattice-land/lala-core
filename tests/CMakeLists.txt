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

file(GLOB test_files src/*.cpp)
foreach(file ${test_files})
  cmake_path(GET file STEM test_name)
  # This condition to ignore GPU tests if GPU is not enabled.
  if(NOT ((${test_name} MATCHES ".*gpu$") AND (NOT GPU)))
    if(GPU)
      set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${test_name} ${file})
    target_include_directories(${test_name} PRIVATE include)
    if(${test_name} MATCHES ".*gpu$")
      target_link_libraries(${test_name} lala_parsing)
      # We run the tests using compute-sanitizer to check for memory leaks.
      add_test(NAME ${test_name} COMMAND compute-sanitizer --error-exitcode 1 $<TARGET_FILE:${test_name}>)
    else()
      target_link_libraries(${test_name} lala_parsing gtest_main)
      gtest_discover_tests(${test_name})
    endif()
  endif()
endforeach()

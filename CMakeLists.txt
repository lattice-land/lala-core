cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

project(lala-core LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# A new configuration mode "LDebug" for light debug.

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES LDebug)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
    "Add a light debug configuration (LDebug)"
    FORCE)
endif()

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Ecuda dependency (header only library)

FetchContent_Declare(
    ecuda
    GIT_REPOSITORY    "https://github.com/lattice-land/ecuda.git"
    GIT_TAG           6a004ad11c7d938b0c247d474a2f733d6aa71499
)

FetchContent_GetProperties(ecuda)
if(NOT ecuda_POPULATED)
    FetchContent_Populate(ecuda)
endif()

if(GPU)
    set_source_files_properties(
        src/ast.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

add_library(lala_core src/ast.cpp)
target_include_directories(lala_core PUBLIC include)
target_include_directories(lala_core PRIVATE ${ecuda_SOURCE_DIR}/include)

if(GPU)
    set_property(TARGET lala_core PROPERTY CUDA_ARCHITECTURES 75)
    set_property(TARGET lala_core APPEND PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_compile_options(lala_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -rdc=true>
    $<$<CONFIG:Debug>:-g -G -DDEBUG>
    $<$<CONFIG:LDebug>:-g -G -DTRACE>
)

enable_testing()

add_executable(
  ast_test
  tests/ast_test.cpp
)
target_link_libraries(
  ast_test
  lala_core
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(ast_test)

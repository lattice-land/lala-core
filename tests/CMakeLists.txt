cmake_minimum_required(VERSION 3.20)

include(FetchContent)
include(CMakePrintHelpers)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
# Fix found at https://github.com/ethereum/solidity/pull/13429
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

project(lala-core LANGUAGES CUDA CXX)
option(GPU "GPU" ON)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
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

# Cuda-battery dependency

FetchContent_Declare(
  cuda_battery
  GIT_REPOSITORY "https://github.com/lattice-land/cuda-battery.git"
  GIT_TAG        develop
)
FetchContent_MakeAvailable(cuda_battery)

# Lala-parsing dependency (just for tests).

FetchContent_Declare(
  lala_parsing
  GIT_REPOSITORY "https://github.com/lattice-land/lala-parsing.git"
  GIT_TAG        develop
)
FetchContent_MakeAvailable(lala_parsing)

# Thrust dependency

FetchContent_Declare(
  thrust
  GIT_REPOSITORY "https://github.com/NVIDIA/thrust.git"
  GIT_TAG        1.12.0
)
FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
    FetchContent_Populate(thrust)
endif()

# Google Test dependency

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(ast_test src/ast_test.cpp)
target_link_libraries(ast_test cuda_battery lala_parsing gtest_main)
target_include_directories(ast_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_parsing_SOURCE_DIR}/include ${thrust_SOURCE_DIR} ${cpp-peglib_SOURCE_DIR})
gtest_discover_tests(ast_test)

add_executable(copy_dag_test src/copy_dag_test.cpp)
target_link_libraries(copy_dag_test cuda_battery gtest_main)
target_include_directories(copy_dag_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${thrust_SOURCE_DIR})
gtest_discover_tests(copy_dag_test)

add_executable(upset_universe_test src/upset_universe_test.cpp)
target_link_libraries(upset_universe_test cuda_battery lala_parsing gtest_main)
target_include_directories(upset_universe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_parsing_SOURCE_DIR}/include ${thrust_SOURCE_DIR} ${cpp-peglib_SOURCE_DIR})
gtest_discover_tests(upset_universe_test)

add_executable(cartesian_product_test src/cartesian_product_test.cpp)
target_link_libraries(cartesian_product_test cuda_battery lala_parsing gtest_main)
target_include_directories(cartesian_product_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_parsing_SOURCE_DIR}/include ${thrust_SOURCE_DIR} ${cpp-peglib_SOURCE_DIR})
gtest_discover_tests(cartesian_product_test)

add_executable(interval_test src/interval_test.cpp)
target_link_libraries(interval_test cuda_battery lala_parsing gtest_main)
target_include_directories(interval_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_parsing_SOURCE_DIR}/include ${thrust_SOURCE_DIR} ${cpp-peglib_SOURCE_DIR})
gtest_discover_tests(interval_test)

add_executable(vstore_test src/vstore_test.cpp)
target_link_libraries(vstore_test cuda_battery lala_parsing gtest_main)
target_include_directories(vstore_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR}/include ${cuda_battery_SOURCE_DIR}/include ${lala_parsing_SOURCE_DIR}/include ${thrust_SOURCE_DIR} ${cpp-peglib_SOURCE_DIR})
gtest_discover_tests(vstore_test)

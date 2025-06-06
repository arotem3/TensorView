cmake_minimum_required(VERSION 3.10)

project(tensor_view
  DESCRIPTION
  "TensorView is a single header template library defining the class TensorView which is used to structure \
  contiguous arrays into tensor-like objects (like numpy's `ndarray`) whose rank (dimension) is known at compile time.\
  The TensorView is a trivial class and only stores the pointer to the start of the array and the shape of the tensor.\
  The TensorView object can also be used in CUDA code by enabling the USE_CUDA option.\
  Since the TensorView object is trivial, it can wrap device pointers and passed by value to a CUDA __global__ function."
  VERSION 1.0
  LANGUAGES CXX
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CUDA "Add CUDA language features so that TensorView can be used in __host__ or __device__ code spaces." OFF)
option(TENSOR_ALWAYS_MUTABLE
  "If enabled, data managed by a TensorView object is writeable even when the object is const qualified. \
  If disabled, the data is only readable when the TensorView is const qualified."
  OFF
)
option(TENSOR_DEBUG "Enables bound checks for indexing into TensorView objects." OFF)

add_library(tensor_view INTERFACE)
target_sources(tensor_view INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/TensorView.hpp)
target_include_directories(tensor_view INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/TensorView)

message("build = ${CMAKE_BUILD_TYPE}")
string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
if ((build_type STREQUAL debug) OR TENSOR_DEBUG)
    target_compile_definitions(tensor_view INTERFACE TENSOR_DEBUG)
endif()

if (USE_CUDA)
  target_compile_definitions(tensor_view INTERFACE TENSOR_USE_CUDA)
  enable_language(CUDA)
endif()

if (TENSOR_ALWAYS_MUTABLE)
  target_compile_definitions(tensor_view INTERFACE TENSOR_ALWAYS_MUTABLE)
endif()

target_compile_options(tensor_view INTERFACE -Wall -Wextra)

install(TARGETS tensor_view EXPORT tensor_view_config)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/TensorView DESTINATION include)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/TensorView.hpp DESTINATION include)
export(EXPORT tensor_view_config
  NAMESPACE tensor_view::
  FILE tensor_view_config.cmake)

enable_testing()

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

foreach(test_file ${TEST_FILES})
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} ${test_file})
  set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
  target_link_libraries(${test_name} tensor_view)
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
cmake_minimum_required(VERSION 3.10)

project(TensorView
  DESCRIPTION "TensorView: A C++ Template Library for Multi-Dimensional Arrays and Views with Optional CUDA Support."
  VERSION 2.0
  LANGUAGES CXX
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TENSOR_USE_CUDA "Enable CUDA support" OFF)
option(TENSOR_USE_64BIT_INDEX "Use 64-bit index type" OFF)

# ------------------------------------------------------------------------------
# Library target
# ------------------------------------------------------------------------------
add_library(TensorView INTERFACE)

target_sources(TensorView
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/TensorView.hpp>
)

target_include_directories(TensorView
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if (TENSOR_USE_64BIT_INDEX)
  target_compile_definitions(TensorView INTERFACE TENSOR_USE_64BIT_INDEX)
endif()

# ------------------------------------------------------------------------------
# CUDA support
# ------------------------------------------------------------------------------
set(TENSORVIEW_PKGCONFIG_LIBS "")

if (TENSOR_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 12 REQUIRED)

  if (CUDAToolkit_FOUND)
    target_compile_definitions(TensorView INTERFACE TENSOR_USE_CUDA)
    target_link_libraries(TensorView INTERFACE CUDA::cudart)
    target_compile_options(TensorView INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)

    set(TENSORVIEW_PKGCONFIG_LIBS "-lcudart")
  endif()
endif()

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# Install full TensorView directory of headers
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/TensorView
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/TensorView.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export interface target
install(
  TARGETS TensorView
  EXPORT TensorViewTargets
)

# ------------------------------------------------------------------------------
# CMake package config
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorViewConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorView
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorViewConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorView
)

install(
  EXPORT TensorViewTargets
  FILE TensorViewTargets.cmake
  NAMESPACE TensorView::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorView
)

# ------------------------------------------------------------------------------
# pkg-config file generation
# ------------------------------------------------------------------------------
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorView.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/TensorView.pc
  @ONLY
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/TensorView.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
enable_testing()

file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

foreach(test_file ${TEST_FILES})
  get_filename_component(test_name ${test_file} NAME_WE)
  add_executable(${test_name} ${test_file})

  if (TENSOR_USE_CUDA)
    set_source_files_properties(${test_file} PROPERTIES LANGUAGE CUDA)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  set_target_properties(${test_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test
  )

  target_link_libraries(${test_name} PRIVATE TensorView)
  target_compile_options(${test_name} PRIVATE -Wall -Wextra)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

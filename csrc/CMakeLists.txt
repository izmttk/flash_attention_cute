cmake_minimum_required(VERSION 3.18)
project(flash_attention_cute VERSION 0.1.0 LANGUAGES CUDA CXX)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(STANDARD 17) # only c++17 is supported?
set(SM_ARCH 80)
set(INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/3rd/cutlass/include"
    "${CMAKE_SOURCE_DIR}/3rd/cutlass/tools/util/include"
)

file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/*.cu" "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/*.cuh" "${CMAKE_SOURCE_DIR}/*.h" "${CMAKE_SOURCE_DIR}/*.hpp")

set(DEFS
    -D__CUDA_NO_HALF_OPERATORS__
    -D__CUDA_NO_HALF_CONVERSIONS__
    -D__CUDA_NO_HALF2_OPERATORS__
)

set(CMAKE_CXX_STANDARD ${STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

# CUDA 12 下 cmake 编译 torch 会出现 Could NOT find nvtx3
# 参见：https://discuss.pytorch.org/t/failed-to-find-nvtoolsext/179635/12
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
  message(STATUS "PyTorch NVTX headers workaround: Yes")
  # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
  if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
      add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
      # ensure that PyTorch is told to use NVTX3 headers
      target_compile_definitions(
          CUDA::nvToolsExt INTERFACE
          TORCH_CUDA_USE_NVTX3
      )
      target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
  endif()
else()
  message(STATUS "PyTorch NVTX headers workaround: No")
endif()
find_package(Torch REQUIRED)

set(LIB_NAME flash_attention)
add_library(${LIB_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${LIB_NAME} ${TORCH_LIBRARIES} Python::Python)
target_include_directories(${LIB_NAME} PRIVATE ${INCLUDE_DIRS})
target_compile_definitions(${LIB_NAME} PRIVATE ${DEFS})
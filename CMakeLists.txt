cmake_minimum_required(VERSION 3.18)
project(flash_attention_cute VERSION 0.1.0 LANGUAGES CUDA CXX)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(STANDARD 17) # only c++17 is supported?
set(SM_ARCH 80)
set(INCLUDE_DIRS
    "${CMAKE_SOURCE_DIR}/3rd/cutlass/include"
    "${CMAKE_SOURCE_DIR}/include"
)
set(SOURCES
    src/flash_attention.cu
    src/flash_attention_api.cpp
)

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

foreach(SOURCE ${SOURCES})
    get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
    add_executable(${SOURCE_NAME} ${SOURCE})
    target_compile_options(${SOURCE_NAME} PRIVATE -lineinfo -expt-relaxed-constexpr)
    target_include_directories(${SOURCE_NAME} PRIVATE ${INCLUDE_DIRS})
    target_compile_definitions(${SOURCE_NAME} PRIVATE ${DEFS})

    target_link_libraries(${SOURCE_NAME} ${TORCH_LIBRARIES} Python::Python)
    target_include_directories(${SOURCE_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

    get_source_file_property(FILE_LANGUAGE ${SOURCE} LANGUAGE)
    if (FILE_LANGUAGE STREQUAL "CUDA" AND MSVC)
        target_link_options(${SOURCE_NAME} PRIVATE /NODEFAULTLIB:LIBCMT)
        target_compile_options(${SOURCE_NAME} PRIVATE "-Xcompiler=\"/utf-8\"")
    endif()
    # add a run command, usage: cmake --build . --target run_filename
    # add_custom_target(run_${SOURCE_NAME}
    #     COMMAND ${SOURCE_NAME}
    #     DEPENDS ${SOURCE_NAME}
    #     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    # )
endforeach()

option(ENABLE_EXAMPLES "Enable examples" ON)
if (ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

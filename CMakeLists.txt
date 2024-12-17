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
    src/sgemm_1.cu
    src/sgemm_2.cu
    src/sgemm_sm70_twostage.cu
)


set(CMAKE_CXX_STANDARD ${STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCH})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(SOURCE ${SOURCES})
    get_filename_component(SOURCE_NAME ${SOURCE} NAME_WE)
    add_executable(${SOURCE_NAME} ${SOURCE})
    target_compile_options(${SOURCE_NAME} PRIVATE -lineinfo -expt-relaxed-constexpr)
    target_include_directories(${SOURCE_NAME} PRIVATE ${INCLUDE_DIRS})
    if (MSVC)
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




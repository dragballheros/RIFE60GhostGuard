cmake_minimum_required(VERSION 3.16)
# The pinned ncnn and glslang trees call cmake_minimum_required(VERSION 2.8.12).
# CMake 4 rejects that unless this floor is set before those calls.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
project(rife_capi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT RIFE_SRC)
    message(FATAL_ERROR "Set RIFE_SRC to rife-ncnn-vulkan/src (the pinned checkout).")
endif()

find_package(Vulkan REQUIRED)

set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_VULKAN_ONLINE_SPIRV ON CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "" FORCE)
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "" FORCE)
set(NCNN_INSTALL_SDK OFF CACHE BOOL "" FORCE)
# Those two options inject GCC -fno-rtti / -fno-exceptions into every consumer,
# which MSVC rejects. Leave RTTI and exceptions at the compiler default.
set(NCNN_DISABLE_RTTI OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)
set(NCNN_INT8 OFF CACHE BOOL "" FORCE)
# Old glslang's ChooseMSVCCRT.cmake breaks on current MSVC.
set(OVERRIDE_MSVCCRT OFF CACHE BOOL "" FORCE)

add_subdirectory(${RIFE_SRC}/ncnn ${CMAKE_BINARY_DIR}/ncnn)

macro(rife_add_shader SHADER_SRC)
    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_COMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.comp.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_COMP_HEADER}
        COMMAND ${CMAKE_COMMAND} -DSHADER_SRC=${RIFE_SRC}/${SHADER_SRC} -DSHADER_COMP_HEADER=${SHADER_COMP_HEADER} -P "${RIFE_SRC}/generate_shader_comp_header.cmake"
        DEPENDS ${RIFE_SRC}/${SHADER_SRC}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_COMP_HEADER})
endmacro()

rife_add_shader(rife_preproc.comp)
rife_add_shader(rife_postproc.comp)
rife_add_shader(rife_preproc_tta.comp)
rife_add_shader(rife_postproc_tta.comp)
rife_add_shader(rife_flow_tta_avg.comp)
rife_add_shader(rife_v2_flow_tta_avg.comp)
rife_add_shader(rife_v4_flow_tta_avg.comp)
rife_add_shader(rife_flow_tta_temporal_avg.comp)
rife_add_shader(rife_v2_flow_tta_temporal_avg.comp)
rife_add_shader(rife_v4_flow_tta_temporal_avg.comp)
rife_add_shader(rife_out_tta_temporal_avg.comp)
rife_add_shader(rife_v4_timestep.comp)
rife_add_shader(rife_v4_timestep_tta.comp)
rife_add_shader(warp.comp)
rife_add_shader(warp_pack4.comp)
rife_add_shader(warp_pack8.comp)

add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

add_library(rife_capi SHARED
    ${CMAKE_CURRENT_LIST_DIR}/rife_capi.cpp
    ${RIFE_SRC}/rife.cpp
    ${RIFE_SRC}/warp.cpp
)
add_dependencies(rife_capi generate-spirv)
target_include_directories(rife_capi PRIVATE
    ${RIFE_SRC}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/ncnn/src
    ${RIFE_SRC}/ncnn/src
)
target_link_libraries(rife_capi PRIVATE ncnn ${Vulkan_LIBRARY})
if(MSVC)
    target_compile_definitions(rife_capi PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(rife_capi PRIVATE /EHsc /utf-8)
endif()
set_target_properties(rife_capi PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

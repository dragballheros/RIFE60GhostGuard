cmake_minimum_required(VERSION 3.16)
project(ort_capi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ORT_INCLUDE OR NOT ORT_LIB)
    message(FATAL_ERROR "Set ORT_INCLUDE and ORT_LIB. Quote -D arguments in PowerShell so -DNAME=value is not split.")
endif()

# The DirectML nupkg also ships win-arm64 and win-x86 import libraries.
# Linking either of those leaves OrtGetApiBase unresolved on an x64 build.
string(REPLACE "\\" "/" ORT_LIB_SLASH "${ORT_LIB}")
string(TOLOWER "${ORT_LIB_SLASH}" ORT_LIB_LOWER)
if(NOT ORT_LIB_LOWER MATCHES "/runtimes/win-x64/native/onnxruntime\\.lib$")
    message(FATAL_ERROR "ORT_LIB must be runtimes/win-x64/native/onnxruntime.lib, not ${ORT_LIB}")
endif()

add_library(ort_capi SHARED ${CMAKE_CURRENT_LIST_DIR}/../ort_capi.cpp)
target_include_directories(ort_capi PRIVATE ${ORT_INCLUDE})
target_link_libraries(ort_capi PRIVATE ${ORT_LIB})
if(MSVC)
    target_compile_definitions(ort_capi PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(ort_capi PRIVATE /EHsc /utf-8)
endif()

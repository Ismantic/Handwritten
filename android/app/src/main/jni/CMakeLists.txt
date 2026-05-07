cmake_minimum_required(VERSION 3.22)
project(hccr_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 用第三方目录里的 ncnn android 预编译(从 github releases 下下来的 .a 静态库)
set(NCNN_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/ncnn-android/${ANDROID_ABI})

if(NOT EXISTS ${NCNN_ROOT})
    message(FATAL_ERROR "ncnn-android 找不到: ${NCNN_ROOT}\n请把 ncnn-YYYYMMDD-android-vulkan.zip 解压到 android/third_party/ncnn-android/")
endif()

set(ncnn_DIR ${NCNN_ROOT}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(hccr_jni SHARED hccr_jni.cc)
target_link_libraries(hccr_jni
    PRIVATE
        ncnn
        android
        log
)

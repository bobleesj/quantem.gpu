cmake_minimum_required(VERSION 3.22.1)
project(quantem_gpu_android_compat LANGUAGES C CXX)

# Compatibility build entry: one implementation, unchanged library/target names.
enable_testing()
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../vulkan" vulkan)

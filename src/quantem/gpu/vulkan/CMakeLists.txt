cmake_minimum_required(VERSION 3.22.1)
project(quantem_gpu_android_vulkan LANGUAGES C CXX)

option(QUANTEM_GPU_BUILD_ANDROID_BENCHMARK "Build the Android Vulkan benchmark" ON)
option(QUANTEM_GPU_BUILD_CONTRACT_TESTS "Build portable contract tests" ON)

add_library(quantem_gpu_android_contract STATIC
  src/contract.cpp
  src/packed_detector.cpp
  src/qh5_indexed_source.cpp
  src/reference.cpp
)
target_compile_features(quantem_gpu_android_contract PUBLIC cxx_std_20)
target_include_directories(quantem_gpu_android_contract PUBLIC include)
target_compile_options(quantem_gpu_android_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(QUANTEM_GPU_BUILD_CONTRACT_TESTS)
  add_executable(quantem_gpu_android_packed_detector_tests tests/packed_detector_tests.cpp)
  target_link_libraries(quantem_gpu_android_packed_detector_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_packed_detector_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(QUANTEM_GPU_BUILD_CONTRACT_TESTS AND NOT ANDROID)
  enable_testing()
  add_test(NAME quantem_gpu_android_packed_detector_tests COMMAND quantem_gpu_android_packed_detector_tests)
  add_executable(quantem_gpu_android_packed_detector_reference_tests tests/packed_detector_session_tests.cpp)
  target_compile_definitions(quantem_gpu_android_packed_detector_reference_tests PRIVATE PACKED_DETECTOR_REFERENCE_ONLY)
  target_link_libraries(quantem_gpu_android_packed_detector_reference_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_packed_detector_reference_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_packed_detector_reference_tests COMMAND quantem_gpu_android_packed_detector_reference_tests)
  add_executable(quantem_gpu_android_packed_selected_diffraction_reference_tests tests/packed_selected_diffraction_tests.cpp)
  target_compile_definitions(quantem_gpu_android_packed_selected_diffraction_reference_tests PRIVATE PACKED_DETECTOR_REFERENCE_ONLY)
  target_link_libraries(quantem_gpu_android_packed_selected_diffraction_reference_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_packed_selected_diffraction_reference_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_packed_selected_diffraction_reference_tests COMMAND quantem_gpu_android_packed_selected_diffraction_reference_tests)
  add_executable(quantem_gpu_android_contract_tests tests/contract_tests.cpp)
  target_link_libraries(quantem_gpu_android_contract_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_contract_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_contract_tests COMMAND quantem_gpu_android_contract_tests)

  add_executable(quantem_gpu_android_qh5_tests tests/qh5_indexed_source_tests.cpp)
  target_link_libraries(quantem_gpu_android_qh5_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_qh5_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_qh5_tests COMMAND quantem_gpu_android_qh5_tests)

  add_executable(quantem_gpu_android_selected_diffraction_c_api_tests
    tests/selected_diffraction_c_api_tests.cpp
    src/c_api.cpp
  )
  target_link_libraries(quantem_gpu_android_selected_diffraction_c_api_tests PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_selected_diffraction_c_api_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(
    NAME quantem_gpu_android_selected_diffraction_c_api_tests
    COMMAND quantem_gpu_android_selected_diffraction_c_api_tests
  )

  add_executable(quantem_gpu_android_qh5_decode_check benchmarks/qh5_decode_check.cpp)
  target_link_libraries(quantem_gpu_android_qh5_decode_check PRIVATE quantem_gpu_android_contract)
  target_compile_options(quantem_gpu_android_qh5_decode_check PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_executable(quantem_gpu_android_abi_header_tests tests/abi_header_tests.cpp)
  target_include_directories(quantem_gpu_android_abi_header_tests PRIVATE include)
  target_compile_features(quantem_gpu_android_abi_header_tests PRIVATE cxx_std_20)
  target_compile_options(quantem_gpu_android_abi_header_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_abi_header_tests COMMAND quantem_gpu_android_abi_header_tests)

  add_executable(quantem_gpu_android_c_header_tests tests/c_api_header_test.c)
  target_include_directories(quantem_gpu_android_c_header_tests PRIVATE include)
  target_compile_options(quantem_gpu_android_c_header_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME quantem_gpu_android_c_header_tests COMMAND quantem_gpu_android_c_header_tests)
endif()

if(ANDROID)
  if(NOT QUANTEM_GPU_GLSLC)
    file(GLOB _quantem_gpu_glslc_candidates "${ANDROID_NDK}/shader-tools/*/glslc")
    list(LENGTH _quantem_gpu_glslc_candidates _quantem_gpu_glslc_count)
    if(_quantem_gpu_glslc_count GREATER 0)
      list(GET _quantem_gpu_glslc_candidates 0 QUANTEM_GPU_GLSLC)
    endif()
  endif()
  if(NOT EXISTS "${QUANTEM_GPU_GLSLC}")
    message(FATAL_ERROR "Set QUANTEM_GPU_GLSLC to the Android NDK glslc executable")
  endif()

  set(_quantem_gpu_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
  file(MAKE_DIRECTORY "${_quantem_gpu_generated}")
  add_custom_command(
    OUTPUT "${_quantem_gpu_generated}/scan_products_u8_spirv.inc"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -mfmt=c
      -o "${_quantem_gpu_generated}/scan_products_u8_spirv.inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_products_u8.comp"
    DEPENDS shaders/scan_products_u8.comp
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${_quantem_gpu_generated}/mean_diffraction_u8_spirv.inc"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -mfmt=c
      -o "${_quantem_gpu_generated}/mean_diffraction_u8_spirv.inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mean_diffraction_u8.comp"
    DEPENDS shaders/mean_diffraction_u8.comp
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${_quantem_gpu_generated}/scan_products_u16_spirv.inc"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -mfmt=c
      -o "${_quantem_gpu_generated}/scan_products_u16_spirv.inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/scan_products_u16.comp"
    DEPENDS shaders/scan_products_u16.comp
    VERBATIM
  )
  add_custom_command(
    OUTPUT "${_quantem_gpu_generated}/mean_diffraction_u16_spirv.inc"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -mfmt=c
      -o "${_quantem_gpu_generated}/mean_diffraction_u16_spirv.inc"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mean_diffraction_u16.comp"
    DEPENDS shaders/mean_diffraction_u16.comp
    VERBATIM
  )

  foreach(_qh5_shader qh5_lz4_decode qh5_bitunshuffle_u16 packed_lz4_decode packed_compact_header_validate)
    set(_qh5_output "${_quantem_gpu_generated}/${_qh5_shader}_spirv.inc")
    add_custom_command(
      OUTPUT "${_qh5_output}"
      COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O -mfmt=c
        -o "${_qh5_output}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_qh5_shader}.comp"
      DEPENDS "shaders/${_qh5_shader}.comp"
      VERBATIM
    )
    list(APPEND _quantem_gpu_qh5_shaders "${_qh5_output}")
  endforeach()
  set(_workgroup_lz4_output "${_quantem_gpu_generated}/packed_lz4_decode_workgroup_spirv.inc")
  add_custom_command(
    OUTPUT "${_workgroup_lz4_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_WORKGROUP_SYNC=1 -mfmt=c
      -o "${_workgroup_lz4_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_lz4_decode.comp"
    DEPENDS shaders/packed_lz4_decode.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_qh5_shaders "${_workgroup_lz4_output}")
  set(_leader_lz4_output "${_quantem_gpu_generated}/packed_lz4_decode_leader_spirv.inc")
  add_custom_command(
    OUTPUT "${_leader_lz4_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_LEADER_PARSE=1 -mfmt=c
      -o "${_leader_lz4_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_lz4_decode.comp"
    DEPENDS shaders/packed_lz4_decode.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_qh5_shaders "${_leader_lz4_output}")
  set(_scalar_lz4_output "${_quantem_gpu_generated}/packed_lz4_decode_scalar_spirv.inc")
  add_custom_command(
    OUTPUT "${_scalar_lz4_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_SCALAR_CHUNKS=1 -mfmt=c
      -o "${_scalar_lz4_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_lz4_decode.comp"
    DEPENDS shaders/packed_lz4_decode.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_qh5_shaders "${_scalar_lz4_output}")

  foreach(_radial_shader packed_detector_update packed_selected_diffraction prepared_dpc_com prepared_dpc_mean prepared_dpc_center radial_difference radial_fft_rows radial_fft_columns radial_magnitude)
    set(_radial_output "${_quantem_gpu_generated}/${_radial_shader}_spirv.inc")
    add_custom_command(
      OUTPUT "${_radial_output}"
      COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O -mfmt=c
        -o "${_radial_output}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_radial_shader}.comp"
      DEPENDS "shaders/${_radial_shader}.comp"
      VERBATIM
    )
    list(APPEND _quantem_gpu_radial_shaders "${_radial_output}")
  endforeach()
  set(_atomic_detector_output "${_quantem_gpu_generated}/packed_detector_update_atomic_spirv.inc")
  add_custom_command(
    OUTPUT "${_atomic_detector_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_ATOMIC_PARALLEL=1 -mfmt=c
      -o "${_atomic_detector_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_detector_update.comp"
    DEPENDS shaders/packed_detector_update.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_radial_shaders "${_atomic_detector_output}")
  set(_tile_detector_output "${_quantem_gpu_generated}/packed_detector_update_tile_spirv.inc")
  add_custom_command(
    OUTPUT "${_tile_detector_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_TILE_ONLY=1 -mfmt=c
      -o "${_tile_detector_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_detector_update.comp"
    DEPENDS shaders/packed_detector_update.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_radial_shaders "${_tile_detector_output}")
  set(_scalar_detector_output "${_quantem_gpu_generated}/packed_detector_update_scalar_spirv.inc")
  add_custom_command(
    OUTPUT "${_scalar_detector_output}"
    COMMAND "${QUANTEM_GPU_GLSLC}" --target-env=vulkan1.1 -O
      -DQUANTEM_GPU_SCALAR_ONLY=1 -mfmt=c
      -o "${_scalar_detector_output}"
      "${CMAKE_CURRENT_SOURCE_DIR}/shaders/packed_detector_update.comp"
    DEPENDS shaders/packed_detector_update.comp
    VERBATIM
  )
  list(APPEND _quantem_gpu_radial_shaders "${_scalar_detector_output}")

  add_library(quantem_gpu_android_vulkan STATIC
    src/c_api.cpp
    src/packed_detector_session.cpp
    src/radial_fft.cpp
    src/vulkan_exact_products.cpp
    "${_quantem_gpu_generated}/scan_products_u8_spirv.inc"
    "${_quantem_gpu_generated}/mean_diffraction_u8_spirv.inc"
    "${_quantem_gpu_generated}/scan_products_u16_spirv.inc"
    "${_quantem_gpu_generated}/mean_diffraction_u16_spirv.inc"
    ${_quantem_gpu_qh5_shaders}
    ${_quantem_gpu_radial_shaders}
  )
  target_compile_features(quantem_gpu_android_vulkan PUBLIC cxx_std_20)
  target_include_directories(quantem_gpu_android_vulkan PUBLIC include)
  target_include_directories(quantem_gpu_android_vulkan PRIVATE "${_quantem_gpu_generated}")
  target_link_libraries(quantem_gpu_android_vulkan PUBLIC quantem_gpu_android_contract vulkan log android)
  target_compile_options(quantem_gpu_android_vulkan PRIVATE -Wall -Wextra -Wpedantic -Werror)

  if(QUANTEM_GPU_BUILD_CONTRACT_TESTS)
    add_executable(quantem_gpu_android_packed_admission_tests tests/packed_admission_tests.cpp)
    target_link_libraries(quantem_gpu_android_packed_admission_tests PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem_gpu_android_packed_admission_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(quantem_gpu_android_packed_detector_session_tests tests/packed_detector_session_tests.cpp)
    target_link_libraries(quantem_gpu_android_packed_detector_session_tests PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem_gpu_android_packed_detector_session_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(quantem_gpu_android_packed_selected_diffraction_tests tests/packed_selected_diffraction_tests.cpp)
    target_link_libraries(quantem_gpu_android_packed_selected_diffraction_tests PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem_gpu_android_packed_selected_diffraction_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(quantem_gpu_android_radial_fft_tests tests/radial_fft_tests.cpp)
    target_link_libraries(quantem_gpu_android_radial_fft_tests PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem_gpu_android_radial_fft_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_executable(quantem_gpu_android_selected_diffraction_c_api_tests
      tests/selected_diffraction_c_api_tests.cpp
    )
    target_link_libraries(quantem_gpu_android_selected_diffraction_c_api_tests PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem_gpu_android_selected_diffraction_c_api_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()

  if(QUANTEM_GPU_BUILD_ANDROID_BENCHMARK)
    add_executable(quantem-gpu-vulkan-benchmark benchmarks/vulkan_benchmark.cpp)
    target_link_libraries(quantem-gpu-vulkan-benchmark PRIVATE quantem_gpu_android_vulkan)
    target_compile_options(quantem-gpu-vulkan-benchmark PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()


# Canonical names for new consumers; preserve the installed Android ABI names.
add_library(quantem_gpu_vulkan_contract ALIAS quantem_gpu_android_contract)
if(TARGET quantem_gpu_android_vulkan)
  add_library(quantem_gpu_vulkan ALIAS quantem_gpu_android_vulkan)
endif()

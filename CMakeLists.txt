cmake_minimum_required(VERSION 3.18)
project(autostudio
        VERSION 0.1
        LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 14)

# ###############################################################################
# # CUDA compiler setup
# ###############################################################################

# # Libtorch
# set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${CMAKE_SOURCE_DIR}/External/libtorch/)
# find_package(Torch REQUIRED PATHS ${CMAKE_SOURCE_DIR}/External/libtorch NO_DEFAULT_PATH)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# link_directories(${CMAKE_SOURCE_DIR}/External/libtorch/lib/)
# include_directories(${TORCH_INCLUDE_DIRS})

###############################################################################
# CUDA compiler setup
###############################################################################
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

add_definitions(-DHALF_PRECISION)

# tiny-cuda-nn
add_definitions(-DJSON_HAS_CPP_14)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(External/tiny-cuda-nn)
include_directories(${TCNN_INCLUDE_DIRECTORIES})
include_directories(External/tiny-cuda-nn/include)
include_directories(External/tiny-cuda-nn/dependencies)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# zlib, for compiling cnpy
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# Libtorch
set(CMAKE_PREFIX_PATH ${DCMAKE_PREFIX_PATH}$ ${CMAKE_SOURCE_DIR}/External/libtorch/)
find_package(Torch REQUIRED PATHS ${CMAKE_SOURCE_DIR}/External/libtorch NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
link_directories(${CMAKE_SOURCE_DIR}/External/libtorch/lib/)
include_directories(${TORCH_INCLUDE_DIRS})

# YAML
add_subdirectory(External/yaml-cpp)
include_directories(External/yaml-cpp/include)

# eigen
include_directories(External/eigen-3.4.0)
include(src/dataset/dataset.cmake)
include(src/geometry/geometry.cmake)
include(src/utils/utils.cmake)
include(src/modules/camera_manager/camera_manager.cmake)
include(src/modules/common/common.cmake)
include(src/modules/fields/fields.cmake)
include(src/pipeline/pipeline.cmake)

list(APPEND SRC_ALL
        ${SRC_DATASET}
        ${SRC_UTILS})

set(ALL_LIBS ${TORCH_LIBRARIES} ${TORCH_CUDA_LIBRARIES} ${ZLIB_LIBRARIES} yaml-cpp fmt::fmt)
set(ALL_LIBS ${ALL_LIBS} tiny-cuda-nn stdc++fs)

# ------------------------------------ Add executables ---------------------------------------
add_executable(main main.cpp ${SRC_ALL})
target_link_libraries(main ${ALL_LIBS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

# Install scripts
configure_file(scripts/tmp_run.py ${CMAKE_BINARY_DIR}/tmp_run.py COPYONLY)
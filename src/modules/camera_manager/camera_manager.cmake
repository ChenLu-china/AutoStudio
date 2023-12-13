set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/modules/camera_manager)
list(APPEND SRC_DATASET
        ${CUR_DIR}/Camera.cpp
        ${CUR_DIR}/Image.cpp
        ${CUR_DIR}/Image.cu
        ${CUR_DIR}/Sampler.cpp
        ${CUR_DIR}/Sampler.cu
        )
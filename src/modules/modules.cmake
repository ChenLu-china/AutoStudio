set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/modules)
list(APPEND SRC_DATASET
        ${CUR_DIR}/camera_manager/camera.cpp
        ${CUR_DIR}/camera_manager/image.cpp
        ${CUR_DIR}/camera_manager/image.cu
        ${CUR_DIR}/camera_manager/sampler.cpp
        )


set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/modules/common)
list(APPEND SRC_DATASET
        ${CUR_DIR}/BaseModel.cpp
        ${CUR_DIR}/mlp/TinyMLP.cpp
        ${CUR_DIR}/shader/SHShader.cpp
        ${CUR_DIR}/shader/SHShader.cu
    )

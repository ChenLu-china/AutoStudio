set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/utils)
list(APPEND SRC_UTILS
        ${CUR_DIR}/GlobalData.cpp
        ${CUR_DIR}/cnpy.cpp
        ${CUR_DIR}/Utils.cpp
        ${CUR_DIR}/CustomOps/CustomOps.cpp
        ${CUR_DIR}/CustomOps/CustomOps.cu
        ${CUR_DIR}/CustomOps/FlexOps.cpp
        ${CUR_DIR}/CustomOps/FlexOps.cu
        ${CUR_DIR}/CustomOps/Scatter.cpp
        ${CUR_DIR}/CustomOps/Scatter.cu
        )


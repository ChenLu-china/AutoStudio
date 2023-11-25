set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/modules/fields)
list(APPEND SRC_DATASET
        ${CUR_DIR}/FieldsFactory.cpp
        ${CUR_DIR}/HashMap.cpp
        ${CUR_DIR}/HashMap.cu
        ${CUR_DIR}/f2nerf/OctreeMap.cpp
        ${CUR_DIR}/f2nerf/OctreeMap.cu
        ${CUR_DIR}/f2nerf/Octree.cpp
        ${CUR_DIR}/f2nerf/Octree.cu
        ${CUR_DIR}/streetsurf/SSFNGPMap.cpp
        ${CUR_DIR}/streetsurf/SSFNGPMap.cu
        ${CUR_DIR}/ngp/NGPMap.cpp
        ${CUR_DIR}/ngp/NGPMap.cu
        )
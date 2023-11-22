set(CUR_DIR ${PROJECT_SOURCE_DIR}/src/modules)
list(APPEND SRC_DATASET
        ${CUR_DIR}/camera_manager/Camera.cpp
        ${CUR_DIR}/camera_manager/Image.cpp
        ${CUR_DIR}/camera_manager/Image.cu
        ${CUR_DIR}/camera_manager/Sampler.cpp
        ${CUR_DIR}/common/BaseModel.cpp
        ${CUR_DIR}/common/TinyMLP.cpp
        ${CUR_DIR}/fields/FieldsFactory.cpp
        ${CUR_DIR}/fields/HashMap.cpp
        ${CUR_DIR}/fields/HashMap.cu
        ${CUR_DIR}/fields/f2nerf/OctreeMap.cpp
        ${CUR_DIR}/fields/f2nerf/OctreeMap.cu
        ${CUR_DIR}/fields/f2nerf/Octree.cpp
        ${CUR_DIR}/fields/f2nerf/Octree.cu
        ${CUR_DIR}/fields/streetsurf/SSFNGPMap.cpp
        ${CUR_DIR}/fields/streetsurf/SSFNGPMap.cu
        ${CUR_DIR}/fields/ngp/NGPMap.cpp
        ${CUR_DIR}/fields/ngp/NGPMap.cu
        )


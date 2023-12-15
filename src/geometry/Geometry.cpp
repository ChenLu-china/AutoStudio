/**
* This file is part of auto_studio
* Copyright (C) 
*  @file Geometry.cpp
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#ifdef snprintf
#undef snprintf
#endif

#include "MeshExtractor.h"

namespace AutoStudio
{

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<MeshExtractor>(m, "MeshExtractor")
        .def(py::init<FieldModel*>())
        .def("GetGridSamples", &MeshExtractor::GetGridSamples);
    
}  
} // namespace AutoStudio

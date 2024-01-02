/**
* This file is part of auto_studio
* Copyright (C) 
*  @file MeshExtractor.cpp
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/

#include "MeshFactory.h"
#include "MeshExtractor.h"

namespace AutoStudio
{

MeshFactory::MeshFactory(GlobalData* global_data)
{
    global_data_ = global_data;
    std::cout << "Doing MeshExtractor" << std::endl;
    auto conf = global_data_->config_["mesh"];
    const std::string mesh_dtype = conf["meshtype"].as<std::string>();
    if (mesh_dtype == "vis_mesh"){
        mesh_dtype_ = MeshDType::VIS;
    }
    else if (mesh_dtype == "occ_mesh")
    {
        mesh_dtype_ = MeshDType::OCC;
    }
}

std::unique_ptr<MeshExtractor> MeshFactory::CreateMeshExtractor()
{
    if (mesh_dtype_ == 0){
        return std::make_unique<AutoStudio::VisionMeshExtractor>(global_data_);
    }
    else if (mesh_dtype_ == 1)
    {
        return std::make_unique<AutoStudio::OccMeshExtractor>(global_data_);
    }
    return nullptr;
}
// SampleResultFlex MeshExtractor::GetGridSamples(const Tensor& rays_o_raw, const Tensor& rays_d_raw, const Tensor& bounds_raw)
// {
    
// }

} // namespace AutoStudio
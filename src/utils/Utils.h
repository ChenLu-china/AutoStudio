/**
* This file is part of autostudio
* Copyright (C) 
**/


#include <string>
#include <torch/torch.h>

namespace AutoStudio {
using Tensor = torch::Tensor;

void TensorExportPCD(const std::string& path, Tensor verts);
void TensorExportPCD(const std::string& path, Tensor verts, Tensor vert_colors);
int SaveVectorAsNpy(const std::string& path, std::vector<float> data);

} // namespace AutoStudio
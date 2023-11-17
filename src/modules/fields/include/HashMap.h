/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef HASHMAP_H
#define HASHMAP_H
#include <torch/torch.h>
#include "../../../Common.h"
#include "../../../utils/GlobalData.h"
#include "../../common/include/TinyMLP.h"
#include "../../common/include/FieldModel.h"

namespace AutoStudio
{

#define N_CHANNELS 2
#define N_LEVELS 16  // arranged into L levels in instant-ngp

class Hash3DVertex : public FieldModel
{
private:

public:
    Hash3DVertex(GlobalData* global_data);

    std::vector<Tensor> States() override;
    int LoadStates(const std::vector<Tensor>& states, int idx) override;

    int pool_size_;
    int mlp_hidden_dim_, mlp_out_dim_, n_hidden_layers_;
    
    Tensor feat_pool_;
    Tensor prim_pool_;
    Tensor bias_pool_;
    Tensor feat_local_idx_;
    Tensor feat_local_size_;
    
    std::unique_ptr<TMLP> mlp_;
    int n_volumes_;
};

} // namespace AutoStudio

#endif // HASHMAP_H
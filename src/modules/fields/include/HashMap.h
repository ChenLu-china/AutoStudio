/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef HASHMAP_H
#define HASHMAP_H
#include <torch/torch.h>
#include "FieldModel.h"
#include "../../../utils/GlobalData.h"


namespace AutoStudio
{

class Hash3DVertex : public FieldModel
{
private:

public:
    Hash3DVertex(GlobalData* global_data);
};

} // namespace AutoStudio

#endif // HASHMAP_H
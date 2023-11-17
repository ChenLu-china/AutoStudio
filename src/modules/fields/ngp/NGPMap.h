/**
* This file is part of autostudio
* Copyright (C) 
**/


#ifndef NGPMAP_H
#define NGPMAP_H
#include <torch/torch.h>
#include "../../../Common.h"
#include "../../../dataset/Dataset.h"
#include "../../common/include/FieldModel.h"

namespace AutoStudio
{

class NGPMap : public FieldModel
{

private:
    /* data */
public:
    NGPMap(GlobalData* global_data);
};


} // namespace AutoStudio

#endif // NGPMAP_H
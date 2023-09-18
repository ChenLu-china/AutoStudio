/**
* This file is part of auto_studio
* Copyright (C) 
**/


#include "GlobalData.h"


GlobalData::GlobalData(const std::string &config_file)
{
    config_ = YAML::LoadFile(config_file);
}
/**
* This file is part of auto_studio
* Copyright (C) 
**/


#include "GlobalData.h"


namespace Auto_Studio
{

GlobalData::GlobalData(const std::string &config_file)
{
    config_ = YAML::LoadFile(config_file);
}

} // namespace Auto_Studio
/**
* This file is part of auto_studio
* Copyright (C) 
**/


#include "GlobalData.h"


namespace AutoStudio
{

GlobalData::GlobalData(const std::string &config_file)
{
    config_ = YAML::LoadFile(config_file);
    learning_rate_ = config_["train"]["learning_rate"].as<float>();
}

} // namespace AutoStudio
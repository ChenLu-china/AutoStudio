/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef GLOBALDATA_H
#define GLOBALDATA_H

#include <string>
#include <yaml-cpp/yaml.h>


namespace AutoStudio
{

class GlobalData
{
public:
    GlobalData(const std::string &config_file);

    YAML::Node config_;
    std::string base_exp_dir_;

    void *dataset_;
};

} // namespace AutoStudio

#endif // GLOBALDATA_H
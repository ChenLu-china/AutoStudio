/**
* This file is part of auto_studio
* Copyright (C) 
**/


#ifndef GLOBALDATA_H
#define GLOBALDATA_H

#include <string>
#include <yaml-cpp/yaml.h>


namespace Auto_Studio
{

class GlobalData
{
public:
    GlobalData(const std::string &config_file);

public:
    YAML::Node config_;

    void *dataset_;
};

} // namespace Auto_Studio

#endif // GLOBALDATA_H
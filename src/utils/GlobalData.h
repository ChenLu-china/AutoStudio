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

enum RunningMode { TRAIN, VALIDATE };

class GlobalData
{
public:
    GlobalData(const std::string &config_file);

    YAML::Node config_;
    RunningMode mode_;
    std::string base_exp_dir_;

    void *dataset_;


    int n_volumes_ = 1;
    int iter_step_;
    float sampled_oct_per_ray_ = 16.f;
    float sampled_pts_per_ray_ = 512.f;
    float meaningful_sampled_pts_per_ray_ = 512.f;
    float learning_rate_ = 1.f;
    float distortion_weight_ = 0.f;
    float ray_march_fineness_ = 1.f;
    float near_ = 0.1f;
    float gradient_scaling_progress_ = 1.f;
    bool backward_nan_ = false;
};

} // namespace AutoStudio

#endif // GLOBALDATA_H
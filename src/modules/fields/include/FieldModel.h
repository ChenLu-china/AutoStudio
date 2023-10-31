/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   FieldModel.h
*  @author LuChen, 
*  @brief 
*/


#ifndef FIELDMODEL_H
#define FIELDMODEL_H
#include <torch/torch.h>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "../../common/include/BaseModel.h"
#include "../../../utils/GlobalData.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

struct SampleResultFlex {
  Tensor pts;                           // [ n_all_pts, 3 ]
  Tensor dirs;                          // [ n_all_pts, 3 ]
  Tensor dt;                            // [ n_all_pts, 1 ]
  Tensor t;                             // [ n_all_pts, 1 ]
  Tensor anchors;                       // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;                // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;                 // [ n_rays, 1 ]
};


class FieldModel : public BaseModel
{
private:
    /* data */
public:
  virtual Tensor Query(const Tensor& coords) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor Query(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  virtual Tensor AnchoredQuery(const Tensor& coords, const Tensor& anchors) {
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  /*----------------------------------------------Get Points for all-----------------------------------------------------------*/
    
  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }

  virtual SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };
  }
    
    
  /*----------------------------------------Sample Point function for Octree---------------------------------------------------*/
    
  virtual std::tuple<Tensor, Tensor> GetEdgeSamples(int n_pts) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor() };
  }

  virtual void UpdateOctNodes(const SampleResultFlex& sample_result,
                                const Tensor& sampled_weights,
                                const Tensor& sampled_alpha) {
    CHECK(false) << "Not implemented";
  }
    
    
  /*----------------------------------------Sample Point function for Streetsurf-----------------------------------------------*/


  GlobalData* global_data_;
};


} // namespace AutoStudio

#endif // FIELDMODEL_H
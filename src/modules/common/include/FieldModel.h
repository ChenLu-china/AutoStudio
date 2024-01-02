/**
* This file is part of auto_studio
* Copyright (C) 
*  @file   FieldModel.h
*  @author LuChen, ZhenJun Zhao
*  @brief 
*/


#ifndef FIELDMODEL_H
#define FIELDMODEL_H
#include <torch/torch.h>
#include <memory>
#include <yaml-cpp/yaml.h>
#include "BaseModel.h"
#include "../../../utils/GlobalData.h"


namespace AutoStudio
{

using Tensor = torch::Tensor;

struct SampleResultFlex
{
  Tensor pts;                           // [ n_all_pts, 3 ]
  Tensor dirs;                          // [ n_all_pts, 3 ]
  Tensor dt;                            // [ n_all_pts, 1 ]
  Tensor t;                             // [ n_all_pts, 1 ]
  Tensor anchors;                       // [ n_all_pts, 3 ]
  Tensor pts_idx_bounds;                // [ n_rays, 2 ] // start, end
  Tensor first_oct_dis;                 // [ n_rays, 1 ]
};

struct RenderResult
{
    Tensor colors;
    Tensor first_oct_dis;
    Tensor disparity;
    Tensor edge_feats;
    Tensor depth;
    Tensor weights;
    Tensor idx_start_end;
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

  virtual RenderResult Render(const Tensor& rays_o, const Tensor& rays_d, 
                            const Tensor& ranges, const Tensor& emb_idx) {
    CHECK(false) << "Not implemented";
    return { Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor() };                         
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
  /*----------------------------------------Sample Point with density for Octree---------------------------------------------------*/
  
  virtual Tensor GetVisDensities(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds){
    CHECK(false) << "Not implemented";
    return Tensor();
  }

  /*----------------------------------------Sample Point function for Streetsurf-----------------------------------------------*/

  enum BGColorType { white, black, rand_noise };

  GlobalData* global_data_;
};

} // namespace AutoStudio

#endif // FIELDMODEL_H
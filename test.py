import timm
import torch

model = timm.create_model("vit_base_resnet50_384", pretrained=True, pretrained_cfg_overlay=dict(file="/opt/data/private/chenlu/AutoStudio/AutoStudio/pretrained_models/models--timm--vit_base_r50_s16_384.orig_in21k_ft_in1k/pytorch_model.bin"))
# 
# This file is part of autostudio
# Copyright (C) 
#


from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def read_image(path):
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def nearest_patch_multiplier(h, w, patch_size):
    return int(np.round(h / patch_size) * patch_size), int(
        np.round(w / patch_size) * patch_size
    )


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            sample["mask"] = cv2.resize(
                sample["mask"].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample["mask"] = sample["mask"].astype(bool)

        return sample


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True, # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, num_channels=1, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
           self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)


class OmnidataModel:
    ### consts, do not modify ###
    backbone = "vitb_rn50_384"
    patch_size = 32
    channel_dict = {"depth": 1, "normal": 3}
    ckpt_dict = {
        "depth": "omnidata_dpt_depth_v2.ckpt",
        "normal": "omnidata_dpt_normal_v2.ckpt",
    }

    def __init__(self, task="depth", model_path=None, device="cuda:0"):
        if model_path is None:
            model_path = Path.cwd() / "pretrained_models" / self.ckpt_dict[task]

        self.model_path = Path(model_path) / self.ckpt_dict[task]
        self.task = task
        self.channel = self.channel_dict[task]
        self.device = device

        self.model = DPTDepthModel(backbone=self.backbone, num_channels=self.channel)

        checkpoint = torch.load(self.model_path, map_location=device)
        assert "state_dict" in checkpoint, "No state_dict found in checkpoint"

        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            # remove the "model." prefix
            state_dict[k[len("model.") :]] = v
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        im2tensor = [transforms.ToTensor()]
        if task == "depth":
            im2tensor.append(transforms.Normalize(mean=0.5, std=0.5))
        self.im2tensor = transforms.Compose(im2tensor)

    def raw_image_to_tensor(self, im_raw, down_factor):
        # Round to multiplier of 32
        h_raw, w_raw, _ = im_raw.shape
        h_net, w_net = nearest_patch_multiplier(
            h_raw // down_factor, w_raw // down_factor, self.patch_size
        )

        if h_net != h_raw or w_net != w_raw:
            resizer = Resize(
                h_net,
                w_net,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=self.patch_size,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
            im = resizer({"image": im_raw})["image"]
        else:
            im = im_raw

        im_tensor = self.im2tensor(im)
        im_tensor = im_tensor.unsqueeze(0).float().to(self.device)

        return im_tensor

    def tensor_to_image(self, im_tensor, h_raw, w_raw):
        im_tensor = im_tensor.squeeze()

        # Depth
        if im_tensor.ndim == 2:
            im_tensor = im_tensor.unsqueeze(dim=0)

        _, h_net, w_net = im_tensor.shape
        if h_net != h_raw or w_net != w_raw:
            # See https://github.com/isl-org/DPT/blob/main/run_monodepth.py
            im_tensor = torch.nn.functional.interpolate(
                im_tensor.unsqueeze(0),
                size=(h_raw, w_raw),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        return im_tensor.cpu().numpy().transpose(1, 2, 0)

    def __call__(self, im_fname, down_factor=1):
        im_raw = read_image(str(im_fname))
        h_raw, w_raw, _ = im_raw.shape

        im_tensor = self.raw_image_to_tensor(im_raw, down_factor=down_factor)

        # Feed into network
        with torch.no_grad():
            output = self.model(im_tensor)

        # Resize back to original size
        output = self.tensor_to_image(output, h_raw, w_raw)
        return output
import os
import cv2
import sys
import PIL
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, field

from ..modules import DPTDepthModel
from ..modules import get_transform
from ..modules.omnidata_tools import UNet


@dataclass
class OmnidataConfig:
    
    def __init__(self, args=None) -> None:
        
        if args is None:
            args = self.init_parsers()
        
        self.omnidata_path = args.omnidata_path
        self.pretrained_models = args.pretrained_models
        self.task = args.task
        self.img_path = args.img_path
        self.output_path = args.output_path
        self.patch_size = args.patch_size

    def init_parsers(self):

        parser = argparse.ArgumentParser(description="generate depth or surface noramls use Omnidata fucntion")

        parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model")
        parser.set_defaults(omnidata_path='/home/yuzh/Projects/omnidata/omnidata_tools/torch/')

        parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
        parser.set_defaults(pretrained_models='/opt/data/private/chenlu/AutoStudio/pretrained_models/pretrained_omnidata')

        parser.add_argument('--task', dest='task', help="normal or depth")
        parser.set_defaults(task='normal')

        parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
        # parser.set_defaults(im_name='NONE')

        parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
        # parser.set_defaults(store_name='')

        parser.add_argument('--patch_size', type=int, default=32, help="")
        args = parser.parse_args()

        return args


@dataclass
class Omnidata():

    config=OmnidataConfig()

    def __init__(self, task, pretrain_models:str) -> None:
        self.root_dir = pretrain_models
        self.omnidata_path = self.config.omnidata_path
        self.config.task = task
        self.patch_size = self.config.patch_size
        self.model, self.trans_totensor = self._generate()

    def _load_normal_module(self, map_location, image_size:int =384, device=torch.device("cpu")):
        pertrained_weight_path = os.path.join(self.root_dir, "omnidata_dpt_normal_v2.ckpt")
        model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3) 
        checkpoint = torch.load(pertrained_weight_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(image_size),
                get_transform("rgb", image_size=None),
            ]
        )
        return model, trans_totensor
    
    def _load_depth_module(self, map_location, image_size:int =384, device=torch.device("cpu")):
        pretrained_weights_path = os.path.join(self.root_dir, "omnidata_dpt_depth_v2.ckpt")  # 'omnidata_dpt_depth_v1.ckpt'
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        return model, trans_totensor


    def _generate(self):
        
        self.trans_topil = transforms.ToPILImage()
        # os.system(f"mkdir -p {self.config.output_path}")
        self.map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image_size = 384

        if self.config.task == 'normal':
            im2tensor = [transforms.ToTensor()]
            model, trans_totensor = self._load_normal_module(self.map_location, image_size, self.device)
        
        elif self.config.task == 'depth':
            im2tensor = [transforms.ToTensor()]
            im2tensor.append(transforms.Normalize(mean=0.5, std=0.5))
            model, trans_totensor = self._load_depth_module(self.map_location, image_size, self.device)
        else:
            print("task should be one of the following: normal, depth")
            sys.exit()

        
        trans_rgb = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                transforms.CenterCrop(image_size),
            ]
        )
        self.im2tensor = transforms.Compose(im2tensor)
        return model, trans_totensor

    @staticmethod
    def nearest_patch_multiplier(h, w, patch_size):
        return int(np.round(h / patch_size) * patch_size), int(
            np.round(w / patch_size) * patch_size
        )

    def raw_image_to_tensor(self, im_raw, down_factor):
        # Round to multiplier of 32
        h_raw, w_raw, _ = im_raw.shape
        h_net, w_net = self.nearest_patch_multiplier(
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

    @staticmethod
    def read_image(path):
        img = cv2.imread(path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        return img


    def __call__(self, im_fname, down_factor=1):
        im_raw = self.read_image(str(im_fname))
        h_raw, w_raw, _ = im_raw.shape

        im_tensor = self.raw_image_to_tensor(im_raw, down_factor=down_factor)

        # Feed into network
        with torch.no_grad():
            output = self.model(im_tensor)

        # Resize back to original size
        output = self.tensor_to_image(output, h_raw, w_raw)
        return output
            


@dataclass
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
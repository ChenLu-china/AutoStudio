import cv2,torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def colorize(image, cmap="turbo"):
    h, w, c = image.shape
    # print(h, w, c)
    if c == 1:  # depth
        image = image.squeeze()
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        cmap = plt.get_cmap(cmap)
        image_colorized = cmap(image_normalized)[:, :, :3]
        return np.uint8(image_colorized * 255)
    else:
        return np.uint8(image * 255)


def post_prediction(output, image_fname, output_vis_path=None, output_npy_path=None):
    if output_vis_path is not None:
        output_vis = colorize(output)
        plt.imsave(output_vis_path / f"{image_fname.stem}.png", output_vis)
    if output_npy_path is not None:
        np.save(output_npy_path / f"{image_fname.stem}.npy", output)
    # if args.plt_vis:
    #     plt.imshow(colorize(output))
    #     plt.show()


def visualize_depth(depth, cmap=cv2.COLORMAP_TWILIGHT_SHIFTED):
    """
    depth: (H, W)
    """
    x = depth
    
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    return x_

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]
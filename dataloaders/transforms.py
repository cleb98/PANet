"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.resize(img, self.size, interpolation=Image.BILINEAR)
        # if isinstance(label, dict):
        #     label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
        #              for catId, x in label.items()}
        # else:
        #     label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        if isinstance(label, dict):
            label = {
                catId: self._resize_tensor(x, interpolation=Image.NEAREST)
                for catId, x in label.items()
            }
        else:
            label = self._resize_tensor(label,
                                        interpolation=Image.NEAREST)
        # inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        # scribble = tr_F.resize(scribble, self.size, interpolation=Image.LANCZOS) #prima c'era anti-aliasing
            # For instance and scribble masks (they are tensors)
        inst = self._resize_tensor(inst,
                                   interpolation=Image.NEAREST)
        scribble = self._resize_tensor(scribble,
                                       interpolation=Image.BILINEAR)  # if "lanczos" causes issues, try "bilinear"

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

    def _resize_tensor(self, x, interpolation):
        # If x is a 2D tensor (mask), add a channel dimension.
        if torch.is_tensor(x) and x.dim() == 2:
            x = x.unsqueeze(0)  # from (H, W) to (1, H, W)
            x = tr_F.resize(x, self.size, interpolation=interpolation)
            # x = x.squeeze(0)    # back to (H, W)
        else:
            x = tr_F.resize(x, self.size, interpolation=interpolation)
        return x

class DilateScribble(object):
    """
    Dilate the scribble mask

    Args:
        size: window width
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        scribble = sample['scribble']
        dilated_scribble = Image.fromarray(
            ndimage.minimum_filter(np.array(scribble), size=self.size))
        dilated_scribble.putpalette(scribble.getpalette())

        sample['scribble'] = dilated_scribble
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
        if inst is not None :
            inst = torch.Tensor(np.array(inst)).long()
        if scribble is not None:
            scribble = torch.Tensor(np.array(scribble)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

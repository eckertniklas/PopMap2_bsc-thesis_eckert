# Data Augmentations

import os
import random

import torch
import torchvision.transforms.functional as TF
from torch import Tensor
from utils.constants import config_path
from utils.file_folder_ops import load_json

from utils.utils import *
from utils.plot import plot_2dmatrix


class OwnCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        for t in self.transforms:
            x = t(x)
        return x if mask is None else (x, mask)
    

class AddGaussianNoise(object):
    """Add gaussian noise to a tensor image with a given probability.
    Args:
        mean (float): mean of the noise distribution. Default value is 0.
        std (float): standard deviation of the noise distribution. Default value is 1.
        p (float): probability of the noise beeing applied. Default value is 1.0.
    """
    def __init__(self, mean=0., std=1., p=1.0):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            # x += torch.randn(x.size()) * self.std + self.mean
            x += torch.randn_like(x) * self.std + self.mean
        return x if mask is None else (x, mask)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class RandomHorizontalVerticalFlip(object):
    def __init__(self, p=0.5): 
        self.p = p
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            return TF.hflip(TF.vflip(x)) if mask is None else (TF.hflip(TF.vflip(x)), TF.hflip(TF.vflip(mask)))
        return x if mask is None else (x, mask)
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5): 
        self.p = p
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            return TF.vflip(x) if mask is None else (TF.vflip(x), TF.vflip(mask))
        return x if mask is None else (x, mask)
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5): 
        self.p = p
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            return TF.hflip(x) if mask is None else (TF.hflip(x), TF.hflip(mask))
        return x if mask is None else (x, mask)
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomRotationTransform(torch.nn.Module):
    """Rotate by one of the given angles.
    Args:
        angles (sequence): sequence of rotation angles
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, angles, p=0.5):
        self.angles = angles
        self.p = p

    def __call__(self, x):
        """
        Description:
            Rotate the input tensor image by one of the given angles.  
        """
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        if torch.rand(1) < self.p:
            angle = random.choice(self.angles)
            return TF.rotate(x, angle) if mask is None else (TF.rotate(x, angle), TF.rotate(mask, angle))
        return x if mask is None else (x, mask)


class RandomGamma(torch.nn.Module):
    """
    Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    """

    def __init__(self, gamma_limit=(0.5, 2.0), p=0.5):
        self.gamma_limit = gamma_limit
        self.p = p
        self.s2_max = 10000

    def __call__(self, x):
        if torch.rand(1) < self.p:
            gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1])
            x = torch.clip(x, min=0)
            x = x / self.s2_max
            if x.shape[0] == 3:
                x = TF.adjust_gamma(x, gamma)
            else:
                # Apply gamma to each channel separately
                for i in range(x.shape[0]):
                    x[i] = TF.adjust_gamma(x[i], gamma)
            x = x * self.s2_max
        return x


class RandomBrightness(torch.nn.Module):
    """Perform random brightness on an image.
    """

    def __init__(self, beta_limit=(0.666, 1.5), p=0.5):
        self.beta_limit = beta_limit
        self.p = p
        self.s2_max = 10000

    def __call__(self, x):
        """
        Applies the random brightness transformation with probability p.

        :param x: Tensor, input image
        :return: Tensor, output image with brightness adjusted if the transformation was applied
        """
        if torch.rand(1) < self.p:
            beta = random.uniform(self.beta_limit[0], self.beta_limit[1])
            x = x / self.s2_max
            if x.shape[0] == 3:
                x = TF.adjust_brightness(x, beta)
            else:
                # Apply brightness to each channel separately
                for i in range(x.shape[0]):
                    x[i] = TF.adjust_brightness(x[i], beta) 
            x = x * self.s2_max
        return x


# import torch
# import torch.nn as nn
# import cv2
# import numpy as np


# class SyntheticHaze(nn.Module):
#     def __init__(self, p=0.75, haze_intensity=0.5, blur_radius=30, brightness_reduction=40):
#         super(SyntheticHaze, self).__init__()
#         self.p = p
#         self.haze_intensity = haze_intensity
#         self.blur_radius = blur_radius
#         self.brightness_reduction = brightness_reduction
    
#     def create_gaussian_kernel(self, size, sigma=None):
#         if sigma is None:
#             sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
#         x = torch.linspace(-(size // 2), size // 2, steps=size)
#         gauss_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
#         gauss_kernel /= gauss_kernel.sum()
#         return gauss_kernel.view(1, -1)

#     def gaussian_blur(self, img):
#         padding = self.blur_radius // 2
#         img_padded = torch.nn.functional.pad(img, (padding, padding, padding, padding), mode='reflect')
#         # kernel = cv2.getGaussianKernel(self.blur_radius, 0)
#         # kernel = torch.from_numpy(kernel).float().view(1, -1)
#         kernel = self.create_gaussian_kernel(self.blur_radius)
#         kernel2d = torch.matmul(kernel.t(), kernel)
#         # kernel2d = kernel2d.view(1, 1, self.blur_radius, self.blur_radius)
#         kernel2d = kernel2d.repeat(img.shape[0], 1, 1, 1)
#         blurred_img = torch.nn.functional.conv2d(img_padded, kernel2d, groups=img.shape[0])
#         return blurred_img

#     def forward(self, x):
#         if torch.rand(1) < self.p:
#             x_float = x.float()
#             white_img = torch.full_like(x_float, 255)
#             haze_img = self.gaussian_blur(white_img)
#             haze_img -= self.brightness_reduction
#             haze_img = torch.clamp(haze_img, 0, 255)
#             hazed_image  = x_float * (1 - self.haze_intensity) + haze_img * self.haze_intensity
#             hazed_image  = torch.clamp(hazed_image , 0, 255).to(x.dtype)
#             x = hazed_image
#         return x
    

def generate_haze_parameters():
    # Generate random haze parameters based on atmospheric conditions
    atmosphere_light = np.random.uniform(0.7, 1.0) # Atmospheric light intensity
    haze_density = np.random.uniform(0.05, 0.3)
    return atmosphere_light, haze_density

class HazeAdditionModule(torch.nn.Module):
    def __init__(self, atm_limit=(0.3, 1.0), haze_limit=(0.05,0.3), p=0.9):
        super(HazeAdditionModule, self).__init__()
        self.atm_limit = atm_limit
        self.haze_limit = haze_limit
        self.p = p
        self.s2_max = 10000

    def forward(self, x):
        """
        Args:
            x: Multispectral satellite imagery, Tensor of shape (batch_size, num_channels, height, width) or (num_channels, height, width)
        Returns:
            x_haze: Hazy multispectral satellite imagery, Tensor of the same shape as x
        """
        if torch.rand(1) < self.p:
            
            # Check if the input image is 3D or 4D
            if len(x.shape) == 3:
                num_channels, height, width = x.shape
                x = x.unsqueeze(0)  # Add a batch dimension
                batch_size = 1
                batched = False
            else:
                batch_size, num_channels, height, width = x.shape 
                batched = True

            # sample/gernerate params
            atmosphere_light = np.random.uniform(self.atm_limit[0], self.atm_limit[1]) # Sample Atmospheric light intensity
            haze_density = np.random.uniform(self.haze_limit[0], self.haze_limit[1]) # Sample Haze density

            # Create the haze layer
            haze_layer = torch.ones((batch_size, num_channels, height, width), dtype=x.dtype, device=x.device) * atmosphere_light

            # Apply haze to the input image
            x = x / self.s2_max
            x_haze = x * (1 - haze_density) + haze_layer * haze_density
            x_haze = x_haze * self.s2_max

            # squeeze the batch dimension
            if not batched:
                x_haze = x_haze.squeeze(0)  # Remove the batch dimension

            # reassign the output image to the return variable
            x = x_haze

        return x
    
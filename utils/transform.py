# Data Augmentations

import os
import random

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch import Tensor
from utils.constants import config_path
from utils.file_folder_ops import load_json

from utils.utils import *
from utils.plot import plot_2dmatrix
from utils.utils import Namespace

# CycleGAN
from model.cycleGAN.models import create_model



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


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    

class Eu2Rwa(object):
    def __init__(self, model_checkpoint="eu2rwa_cycleganFreeze", p=0.5):
        
        opt = Namespace(model="test", name=model_checkpoint, input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='instance',
                        no_dropout=True, init_type="normal", init_gain=0.02, epoch="latest", load_iter=0, isTrain=False, gpu_ids=[0],
                        preprocess=None,  model_suffix="_A", dataset_mode="single", verbose=False,
                        checkpoints_dir="/scratch2/metzgern/HAC/code/CycleGANAugs/pytorch-CycleGAN-and-pix2pix/checkpoints/" )
        self.model = create_model(opt)      # create a model given opt.model and other options
        self.model.setup(opt)               # regular setup: load and print networks; create schedulers
        self.model.eval()   
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        self.unnormalize  = transforms.Normalize( mean=[-1, -1, -1], std=[2, 2, 2] )            
        self.p = p

    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x

        if torch.rand(1) < self.p:
            selection = torch.rand(x.shape[0])<self.p

            # Preprocess, according to the training protocol
            # x = torch.clip((x[selection]-50)/4000*255, 0, 255)/255 # clip to 0-255
            x_select = torch.clip((x[selection]-50)/4000, 0, 1) # clip to 0-255
            x_select = self.normalize(x_select) # normalize to -1,1

            # CycleGAn expects a dictionary with the key 'A' and 'A_paths'
            # data = {'A': x[selection], 'A_paths': None, 'B': None, 'B_paths': None}
            # data = {'A': x_select, 'A_paths': None}
            self.model.set_input({'A': x_select, 'A_paths': None})  # unpack data from data loader
            self.model.test()           # run inference

            # Postprocess  according to the training protocol
            x_select = self.unnormalize(self.model.fake)*4000+50

            # Replace the selected samples with the processed ones
            x[selection] = x_select # G(real)

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
    def __init__(self, p=0.5, allsame=False): 
        self.p = p
        self.allsame = allsame
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x

        if self.allsame:
            if torch.rand(1) < self.p:
                x = TF.hflip(TF.vflip(x))
                if mask is not None:
                    mask = TF.hflip(TF.vflip(mask))
                    return x, mask
                return x
            else:
                if mask is not None:
                    return x, mask
                return x
        else:
            selection = torch.rand(x.shape[0])<self.p
            x[selection] = TF.hflip(TF.vflip(x))[selection]
            if mask is not None:
                mask[selection] = TF.hflip(TF.vflip(mask))[selection]
                return x, mask
            return x
    
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)



class RandomVerticalFlip(object):
    def __init__(self, p=0.5, allsame=False): 
        self.p = p
        self.allsame = allsame
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        
        if self.allsame:
            if torch.rand(1) < self.p:
                x = TF.vflip(x)
                if mask is not None:
                    mask = TF.vflip(mask)
                    return x, mask
                return x
            else:
                if mask is not None:
                    return x, mask
                return x
        else:
            # random horizontal flip with probability 0.5 for each sample in batch
            selection = torch.rand(x.shape[0])<self.p
            x[selection] = TF.vflip(x)[selection]
            if mask is not None:
                mask[selection] = TF.vflip(mask)[selection]
                return x, mask
            return x 
        
    def __repr__(self):
        return self.__class__.__name__ + '(p={0}'.format(self.p)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, allsame=False): 
        self.p = p
        self.allsame = allsame
        
    def __call__(self, x):
        if torch.is_tensor(x):
            mask = None
        else:
            x, mask = x
        
        if self.allsame:
            if torch.rand(1) < self.p:
                x = TF.hflip(x)
                if mask is not None:
                    mask = TF.hflip(mask)
                    return x, mask
                return x
            else:
                if mask is not None:
                    return x, mask
                return x
        else:
            # random horizontal flip with probability 0.5 for each sample in batch
            selection = torch.rand(x.shape[0])<self.p
            x[selection] = TF.hflip(x)[selection]
            if mask is not None:
                mask[selection] = TF.hflip(mask)[selection]
                return x, mask
            return x
        
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
            if mask is not None:
                return TF.rotate(x, angle, expand=True), TF.rotate(mask, angle, expand=True, fill=-1)
                # return x.permute(0,1,3,2), mask.permute(0,1,3,2)
            return TF.rotate(x, angle)
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

            # convert to 0-1 range
            x = x / self.s2_max

            if len(x.shape) == 3:
                if x.shape[0] == 3:
                    x = TF.adjust_brightness(x, gamma)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[i:i+1] = TF.adjust_gamma(x[i:i+1], gamma)
            elif len(x.shape) == 4:
                if x.shape[1] == 3:
                    x = TF.adjust_brightness(x, gamma)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[:,i:i+1] = TF.adjust_gamma(x[:,i:i+1], gamma)

            # if x.shape[0] == 3:
            #     x = TF.adjust_gamma(x, gamma)
            # else:
            #     # Apply gamma to each channel separately
            #     for i in range(x.shape[0]):
            #         x[:,i] = TF.adjust_gamma(x[:,i], gamma)


            # convert back to 0-10000 range
            x = x * self.s2_max
        return x


class RandomBrightness(torch.nn.Module):
    """Perform random brightness on an image.
    """

    def __init__(self, beta_limit=(0.666, 1.5), p=0.5):
        self.beta_limit = beta_limit
        self.p = p
        self.s2_max = 10000 # for the conversion to a pillow-typical range

    def __call__(self, x):
        """
        Applies the random brightness transformation with probability p.

        :param x: Tensor, input image
        :return: Tensor, output image with brightness adjusted if the transformation was applied
        """
        if torch.rand(1) < self.p:

            # get random brightness factor
            beta = random.uniform(self.beta_limit[0], self.beta_limit[1])

            # convert to pillow-typical range
            x = x / self.s2_max


            if len(x.shape) == 3:
                if x.shape[0] == 3:
                    x = TF.adjust_brightness(x, beta)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[i:i+1] = TF.adjust_brightness(x[i:i+1], beta)
            elif len(x.shape) == 4:
                if x.shape[1] == 3:
                    x = TF.adjust_brightness(x, beta)
                else:
                    # Apply brightness to each channel separately
                    for i in range(x.shape[1]):
                        x[:,i:i+1] = TF.adjust_brightness(x[:,i:i+1], beta)

            # back to the original range
            x = x * self.s2_max
        return x
    

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
    
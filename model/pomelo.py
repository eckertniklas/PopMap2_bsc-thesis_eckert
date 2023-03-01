

import torchvision.models as models
import torch.nn as nn
import torch

# import copy
import segmentation_models_pytorch as smp
from torch.nn.functional import upsample_nearest, interpolate

from utils.utils import plot_2dmatrix


class JacobsUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim):
        super(JacobsUNet, self).__init__()

        ic = input_channels
        
        self.encoder = nn.Sequential(
            nn.Sequential(
                    nn.Conv2d(ic, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(32, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(64, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(128, 256, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(256, 256, kernel_size=3, padding='same'),  nn.Softplus() ) 
        )


        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(384, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(64, 64, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(32, feature_dim, kernel_size=3, padding='same'),  nn.Softplus()),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Sequential(
                    nn.Conv2d(ic, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(32, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(64, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus() ),
        )
        self.decoder2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(64, 64, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(32, feature_dim, kernel_size=3, padding='same'),  nn.Softplus()),
        )

        self.head = nn.Sequential(nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1), nn.Softplus())

        self.unetmodel = nn.Sequential(
            smp.Unet( encoder_name="resnet18", encoder_weights="imagenet", decoder_channels=(64, 32, 16),
                encoder_depth=3, in_channels=input_channels,  classes=feature_dim ),
            nn.Softplus()
        )


    def forward(self, inputs, train=False):

        deactivated = False

        if deactivated:
            p2d = (2, 2, 2, 2)
            x = nn.functional.pad(inputs["input"], p2d, mode='reflect')
            x = self.unetmodel(x)[:,:,2:-2,2:-2]
        else:
                
            s = [ inputs["input"].shape[2]//4, inputs["input"].shape[2]//2, inputs["input"].shape[2] ]
            s2 = [ inputs["input"].shape[2]//2, inputs["input"].shape[2] ]

            x = inputs["input"]
            
            #Encoding
            fmaps = []
            for layer in self.encoder:
                x = layer(x)
                fmaps.append(x)

            # remove this fmap, since it is the same as "x"
            del fmaps[-1]

            # Decoding
            for i, layer in enumerate(self.decoder):
                decodermap = torch.concatenate([ interpolate(x,(s[i],s[i])), fmaps[-(i+1)] ],1)
                x = layer(decodermap)

        x = self.head(x)
        Popmap = x[:,0]
        # Popmap = torch.exp(x[:,0])
        Popcount = Popmap.sum((1,2))

        builtmap = x[:,0]
        builtcount = builtmap.sum((1,2))

        return {"Popcount": Popcount, "Popmap": Popmap, "builtmap": builtmap, "builtcount": builtcount}




class PomeloUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, num_classes):
        super(PomeloUNet, self).__init__()
        
        self.PomeloEncoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding='same'),  nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
        )

        self.PomeloDecoder = nn.Sequential(
            nn.Conv2d(32+1, 128, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(), 
            nn.Conv2d(128, 1, kernel_size=3, padding='same'), nn.SELU(),
        )

        ## set model features
        self.unetmodel = smp.Unet( encoder_name="resnet18", encoder_weights="imagenet",
                                  encoder_depth=3, in_channels=input_channels,  classes=num_classes )
        
        self.gumbeltau = torch.nn.Parameter(torch.tensor([2/3]), requires_grad=True)
        

    def forward(self, inputs):

        #Encoding
        encoding = self.PomeloEncoder(inputs)

        #Decode Buildings
        unetout = self.unetmodel(encoding)
        built_hard = torch.nn.functional.gumbel_softmax(unetout[0], tau=self.gumbeltau, hard=True, eps=1e-10, dim=1)[:,0]
        count = unetout[1]
        
        sparse_buildings = built_hard*count 

        with torch.no_grad():
            no_grad_sparse_buildings = sparse_buildings*1.0
            no_grad_count = count*1.0

        # Decode for OccRate (Population)
        OccRate = self.PomeloDecoder(torch.concatenate([encoding,no_grad_sparse_buildings],1))

        # Get population map and total count
        Popmap = OccRate * sparse_buildings
        Popcount = Popmap.sum()
        builtcount = builtcount
        
        return Popcount, {"Popmap": Popmap, "built_map": sparse_buildings, "builtcount": builtcount}

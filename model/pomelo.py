

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


        self.unetmodel = nn.Sequential(
            smp.Unet( encoder_name="resnet18", encoder_weights="imagenet", decoder_channels=(64, 32, 16),
            # smp.Unet( encoder_name="resnet18", encoder_weights="imagenet", decoder_channels=(64, 32, 16),
                encoder_depth=3, in_channels=input_channels,  classes=feature_dim, decoder_use_batchnorm=False ),
            nn.Softplus()
        )

        self.head = nn.Conv2d(feature_dim, 3, kernel_size=3, padding=1)

                                  
    def forward(self, inputs, train=False):

        deactivated = True

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

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

        # Builtup mask
        builtupmap = torch.sigmoid(x[:,2]) 

        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap}



class PomeloUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim):
        super(PomeloUNet, self).__init__()
        
        self.PomeloEncoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding='same'),  nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(32, feature_dim, kernel_size=1, padding='same'), nn.SELU(),
        )

        self.PomeloDecoder = nn.Sequential(
            nn.Conv2d(feature_dim+1, 128, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(), 
            nn.Conv2d(128, 1, kernel_size=3, padding='same'), nn.SELU(),
        )

        ## set model features
        self.unetmodel = smp.Unet( encoder_name="resnet18", encoder_weights="imagenet",
                                  encoder_depth=3, in_channels=input_channels,  classes=2 )
        
        self.gumbeltau = torch.nn.Parameter(torch.tensor([2/3]), requires_grad=True)
        

    def forward(self, inputs):

        #Encoding
        encoding = self.PomeloEncoder(inputs)

        #Decode Buildings
        x = self.unetmodel(encoding)
        
        # # Builtup mask
        # builtupmap = nn.functional.sigmoid(x[:,2]) 
        builtupmap = torch.nn.functional.gumbel_softmax(x[:,0], tau=self.gumbeltau, hard=True, eps=1e-10, dim=1)[:,0]

        # Building map
        builtdensemap = nn.functional.softplus(x[:,0]) * builtupmap
        builtcount = builtdensemap.sum((1,2))
        

        with torch.no_grad():
            no_grad_sparse_buildings = builtdensemap*1.0
            no_grad_count = builtcount*1.0

        # Decode for OccRate (Population)
        OccRate = self.PomeloDecoder(torch.concatenate([encoding,no_grad_sparse_buildings],1))

        # Get population map and total count
        popdensemap = OccRate * builtdensemap
        popcount = popdensemap.sum()
        builtcount = builtcount
        
        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap, "tau": self.gumbeltau.detach()}
        # return Popcount, {"Popmap": Popmap, "built_map": sparse_buildings, "builtcount": builtcount}
    
        # # Population map
        # popdensemap = nn.functional.softplus(x[:,0])
        # popcount = popdensemap.sum((1,2))

        # # Builtup mask
        # builtupmap = nn.functional.sigmoid(x[:,2]) 

        # return {"popcount": popcount, "popdensemap": popdensemap,
        #         "builtdensemap": builtdensemap, "builtcount": builtcount,
        #         "builtupmap": builtupmap}
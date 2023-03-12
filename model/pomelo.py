

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
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18"):
        super(JacobsUNet, self).__init__()

        self.down = 3
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        self.unetmodel = nn.Sequential(
            # torch.nn.BatchNorm2d(input_channels),
            # torch.nn.Dropout2d(),
            smp.Unet( encoder_name=feature_extractor, encoder_weights="imagenet", decoder_channels=(64, 32, 16),
            # smp.Unet( encoder_name="resnet18", encoder_weights="imagenet", decoder_channels=(64, 32, 16),
                encoder_depth=self.down, in_channels=input_channels,  classes=feature_dim, decoder_use_batchnorm=False ),
            # nn.Softplus()
            nn.ReLU()
        )
        params_sum = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)
        self.mysegmentation_head = nn.Conv2d(16, feature_dim, kernel_size=1, padding=0)

        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)
        # self.gumbeltau = torch.nn.Parameter(torch.tensor([2/3]), requires_grad=True)
        
                                  
    def forward(self, inputs, train=False):

        x  = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')

        # forward and remove padding
        x = self.unetmodel(x)[:,:,self.p:-self.p,self.p:-self.p]

        x = self.head(x)

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

        # Builtup mask
        # builtupmap = torch.sigmoid(x[:,2]) 
        # if train:
        #     # builtupmap = (torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]>0.5).float()
        #     builtupmap = torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]
        #     # builtupmap = torch.nn.functional.gumbel_softmax(x[:,2:4], tau=self.gumbeltau, hard=True, eps=1e-10, dim=1)[:,0]
        # else:
        #     # builtupmap = (torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]>0.5).float()

        builtupmap = torch.sigmoid(x[:,2]) 
        # builtupmap = torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]
        # Sparsify
        # popdensemap = builtupmap * popdensemap


        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap}



# Define a resblock
class Block(nn.Module):
    def __init__(self, dimension, k1=3, k2=1, activation=None):
        super(Block, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dimension, dimension, kernel_size=k1, padding=(k1-1)//2), nn.ReLU(),
            nn.Conv2d(dimension, dimension, kernel_size=k2, padding=(k2-1)//2),
        )
        self.act = activation if activation is not None else nn.ReLU()
    def forward(self, x):
        return self.act(self.net(x) + x)


class ResBlocks(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18"):
        super(ResBlocks, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        k1a = 3
        k1b = 3
        k2 = 1

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            
            # Block(feature_dim, k1=3, k2=1),
            # Block(feature_dim, k1=1, k2=1),
            # nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            # Block(feature_dim, k1=3, k2=1),
            # Block(feature_dim, k1=1, k2=1),
            # nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            # Block(feature_dim, k1=3, k2=1),
            # Block(feature_dim, k1=1, k2=1),
            # nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            Block(feature_dim),
        )
        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)

        params_sum = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # self.mysegmentation_head = nn.Conv2d(16, feature_dim, kernel_size=1, padding=0)
        # self.gumbeltau = torch.nn.Parameter(torch.tensor([2/3]), requires_grad=True)
    
                                  
    def forward(self, inputs, train=False):

        x = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
        x = self.model(x)[:,:,self.p:-self.p,self.p:-self.p]
        x = self.head(x)

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

        builtupmap = torch.sigmoid(x[:,2]) 

        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap}


class ResBlocksDeep(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", k1a=3, k1b=3):
        super(ResBlocksDeep, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)
        k1a = 3
        k1b = 3
        k2 = 1

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            # 1
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 2
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 3
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 4
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 5
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 6
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 7
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # 9 
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # ...
        )
        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)

        params_sum = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
                                  
    def forward(self, inputs, train=False):

        x = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
        x = self.model(x)[:,:,self.p:-self.p,self.p:-self.p]
        x = self.head(x)

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

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
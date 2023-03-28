

import torchvision.models as models
import torch.nn as nn
import torch

# import copy
import segmentation_models_pytorch as smp
from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, ReverseLayerF
from torch.nn.functional import upsample_nearest, interpolate

from utils.utils import plot_2dmatrix


class JacobsUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", classifier="v1", head="v1"):
        super(JacobsUNet, self).__init__()

        self.down = 3
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        # Build the main model
        self.unetmodel = nn.Sequential(
            smp.Unet( encoder_name=feature_extractor, encoder_weights="imagenet", decoder_channels=(64, 32, 16),
                encoder_depth=self.down, in_channels=input_channels,  classes=feature_dim, decoder_use_batchnorm=False),
            nn.ReLU()
        )

        # Build the segmentation head
        if head=="v1":
            self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)
        elif head=="v2":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 4, kernel_size=1, padding=0)
            )
        elif head=="v3":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 100, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(100, 100, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(100, 4, kernel_size=1, padding=0)
            )
        elif head=="v4":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 100, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(100, 4, kernel_size=1, padding=0)
            )

        # Build the domain classifier
        if classifier=="v1":
            self.domain_classifier = DomainClassifier(feature_dim)
        elif classifier=="v2":
            self.domain_classifier = DomainClassifier1x1(feature_dim)
        elif classifier=="v3":
            self.domain_classifier = DomainClassifier_v3(feature_dim)
        elif classifier=="v4":
            self.domain_classifier = DomainClassifier_v4(feature_dim)
        elif classifier=="v5":
            self.domain_classifier = DomainClassifier_v5(feature_dim)
        # self.domain_classifier = DomainClassifier1x1(feature_dim)

        # calculate the number of parameters
        self.params_sum = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1):
        

        # Add padding
        if padding:
            x  = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
            p = self.p
        else:
            x = inputs["input"]
            p = None

        # pad to make sure it is divisible by 32
        if (x.shape[2] % 32) != 0:
            p = (x.shape[2] % 64) //2
            x  = nn.functional.pad(inputs["input"], (p,p,p,p), mode='reflect') 

        # Forward the main model
        features = self.unetmodel(x)

        # revert padding
        if p is not None:
            features = features[:,:,p:-p,p:-p]

        # Foward the segmentation head
        out = self.head(features)

        # Foward the domain classifier
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain = self.domain_classifier(reverse_features)
        
        # Population map
        # popdensemap = nn.functional.softplus(x[:,0])
        popdensemap = nn.functional.relu(out[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(out[:,1])
        builtcount = builtdensemap.sum((1,2))

        # Builtup mask
        # builtupmap = torch.sigmoid(x[:,2]) 
        # if train:
        #     # builtupmap = (torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]>0.5).float()
        #     builtupmap = torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]
        #     # builtupmap = torch.nn.functional.gumbel_softmax(x[:,2:4], tau=self.gumbeltau, hard=True, eps=1e-10, dim=1)[:,0]
        # else:
        #     # builtupmap = (torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]>0.5).float()

        builtupmap = torch.sigmoid(out[:,2]) 
        # builtupmap = torch.nn.functional.softmax(x[:,2:4], dim=1)[:,0]
        # Sparsify
        # popdensemap = builtupmap * popdensemap

        # p = torch.sigmoid(x[:,2]) 


        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap, "domain": domain, "features": features}



# Define a resblock
class Block(nn.Module):
    def __init__(self, dimension, k1=3, k2=1, activation=None):
        super(Block, self).__init__()

        # dim_in = dimension if dim_in is None else dim_in

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
            Block(feature_dim),
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


class UResBlocks(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18"):
        super(UResBlocks, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        k1a = 3
        k1b = 3
        k2 = 1

        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1a, k2=k2)  )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim*2, k1=k1a, k2=k2)  )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(feature_dim*2, feature_dim*2, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim*2, k1=k1a, k2=k2),
            nn.ConvTranspose2d(feature_dim*2, feature_dim*2, kernel_size=2, stride=(2,2), ) )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(feature_dim*4, feature_dim*2, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim*2, k1=k1b, k2=k2),
            nn.ConvTranspose2d(feature_dim*2, feature_dim*2, kernel_size=2, stride=(2,2),) )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(feature_dim + feature_dim*2, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1b, k2=k2)  )

        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)

        params_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
                                  
    def forward(self, inputs, train=False):

        x = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f2b = self.enc3(f2)
        f1b = self.dec2(torch.cat([f2,f2b],1))
        featend = self.dec1(torch.cat([f1,f1b],1))

        #unpad
        featend = featend[:,:,self.p:-self.p,self.p:-self.p]

        x = self.head(featend)

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
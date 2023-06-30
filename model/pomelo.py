

import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp
from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, DomainClassifier_v6, ReverseLayerF
from model.customUNet import CustomUNet
from torch.nn.functional import upsample_nearest, interpolate

from utils.plot import plot_2dmatrix, plot_and_save



class JacobsUNet(nn.Module):
    '''
    PomeloUNet
    Description:
        - UNet with a regression head

    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", classifier="v1", head="v1", down=5):
        super(JacobsUNet, self).__init__()
        """
        Args:
            - input_channels (int): number of input channels
            - feature_dim (int): number of output channels of the feature extractor
            - feature_extractor (str): name of the feature extractor
            - classifier (str): name of the classifier
            - head (str): name of the regression head
            - down (int): number of downsampling steps in the feature extractor
        """

        self.down = down
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        # Build the main model
        self.unetmodel = CustomUNet(feature_extractor, in_channels=input_channels, classes=feature_dim, down=self.down)

        # Define batchnorm layer for the feature extractor
        # self.bn = nn.BatchNorm2d(input_channels)
        # self.maxi = torch.tensor([0,0,0,0,0,0], dtype=torch.float32)

        # Build the regression head
        if head=="v1":
            self.head = nn.Conv2d(feature_dim, 5, kernel_size=1, padding=0)
        elif head=="v2":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 5, kernel_size=1, padding=0)
            )
        elif head=="v3":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 100, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(100, 100, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(100, 5, kernel_size=1, padding=0)
            )
        elif head=="v4":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 100, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(100, 5, kernel_size=1, padding=0)
            )
        elif head=="v5":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 5, kernel_size=1, padding=0)
            )

        self.head.bias.data = 0.75 * torch.ones(5)

        # Build the domain classifier
        # latent_dim = self.unetmodel.latent_dim
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
        elif classifier=="v6":
            self.domain_classifier = DomainClassifier_v6(feature_dim)
        elif classifier=="v7":
            self.domain_classifier = nn.Sequential(nn.Conv2d(feature_dim, 1, kernel_size=1, padding=0), nn.Sigmoid())
        elif classifier=="v8":
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.unetmodel.latent_dim, 100), nn.ReLU(),
                nn.Linear(100, 1),  nn.Sigmoid()
            )
        elif classifier=="v9":
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.unetmodel.latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 1),  nn.Sigmoid()
            )
        elif classifier=="v10":
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.unetmodel.latent_dim, 32), nn.ReLU(),
                nn.Linear(32, 1),  nn.Sigmoid()
            )
        elif classifier=="v11":
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.unetmodel.latent_dim, 8), nn.ReLU(),
                nn.Linear(8, 1),  nn.Sigmoid()
            )
        elif classifier=="v12":
            self.domain_classifier = nn.Sequential(
                nn.Linear(self.unetmodel.latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 64),  nn.ReLU(),
                nn.Linear(64, 1),  nn.Sigmoid()
            )
        else:
            self.domain_classifier = None

        # calculate the number of parameters
        self.params_sum = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """
        data = inputs["input"]

        # Add padding
        data, (px1,px2,py1,py2) = self.add_padding(data, padding)

        # Forward the main model
        features, decoder_features = self.unetmodel(data, return_features=return_features)

        # revert padding
        features = self.revert_padding(features, (px1,px2,py1,py2))


        # Forward the head
        out = self.head(features)

        # Foward the domain classifier
        if self.domain_classifier is not None and return_features and alpha>0:
            reverse_features = ReverseLayerF.apply(decoder_features.unsqueeze(3), alpha) # apply gradient reversal layer
            domain = self.domain_classifier(reverse_features.permute(0,2,3,1).reshape(-1, reverse_features.size(1))).view(reverse_features.size(0),-1)
        else:
            domain = None
        
        # Population map and total count
        popdensemap = nn.functional.relu(out[:,0])
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            popcount = (popdensemap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popcount = popdensemap.sum((1,2))

        popvarmap = nn.functional.softplus(out[:,1])
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            popvar = (popvarmap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popvar = popvarmap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(out[:,2])
        builtcount = builtdensemap.sum((1,2))

        # Builtup mask
        builtupmap = torch.sigmoid(out[:,3])

        return {"popcount": popcount, "popdensemap": popdensemap,
                "popvar": popvar ,"popvarmap": popvarmap, 
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap, "domain": domain, "features": features,
                "decoder_features": decoder_features}


    def add_padding(self, data, force=True):
        # Add padding
        px1,px2,py1,py2 = None, None, None, None
        if force:
            data  = nn.functional.pad(data, self.p2d, mode='reflect')
            px1,px2,py1,py2 = self.p, self.p, self.p, self.p
        else:
            # pad to make sure it is divisible by 32
            if (data.shape[2] % 32) != 0:
                px1 = (64 - data.shape[2] % 64) //2
                px2 = (64 - data.shape[2] % 64) - px1
                # data = nn.functional.pad(data, (px1,0,px2,0), mode='reflect') 
                data = nn.functional.pad(data, (0,0,px1,px2,), mode='reflect') 
            if (data.shape[3] % 32) != 0:
                py1 = (64 - data.shape[3] % 64) //2
                py2 = (64 - data.shape[3] % 64) - py1
                data = nn.functional.pad(data, (py1,py2,0,0), mode='reflect')

        return data, (px1,px2,py1,py2)
    
    def revert_padding(self, data, padding):
        px1,px2,py1,py2 = padding
        if px1 is not None or px2 is not None:
            data = data[:,:,px1:-px2,:]
        if py1 is not None or py2 is not None:
            data = data[:,:,:,py1:-py2]
        return data


# Define a resblock
class Block(nn.Module):
    def __init__(self, dimension, k1=3, k2=1, activation=None, squeeze=False):
        super(Block, self).__init__()

        # dim_in = dimension if dim_in is None else dim_in
        if squeeze:
            s = 2
            self.net = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(dimension, dimension, kernel_size=k1, padding=(k1-1)//2), nn.ReLU(),
                nn.Conv2d(dimension, dimension, kernel_size=k2, padding=(k2-1)//2),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(dimension, dimension, kernel_size=k1, padding=(k1-1)//2), nn.ReLU(),
                nn.Conv2d(dimension, dimension, kernel_size=k2, padding=(k2-1)//2),
            )
        self.act = activation if activation is not None else nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.net(x)
        x += identity
        return self.act(x)
        # return self.act(self.net(x) + x)


class ResBlocks(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim):
        super(ResBlocks, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        k1a = 3
        k1b = 3
        k2 = 3

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            # nn.MaxPool2d(8),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),

            # nn.Upsample(scale_factor=8, mode='bilinear'),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
        )
        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)
        


        a = torch.zeros((1,6,128,128))
        a[0,:,64,64] = 500
        plot_2dmatrix(self.model(a)[0,0])

        params_sum = sum(p.numel() for p in self.model.parameters() if p.requires_grad)    
                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=False):

        # Add padding
        if padding:
            x  = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
            p = self.p
        else:
            x = inputs["input"]
            p = None

        # pad to make sure it is divisible by 32
        if (x.shape[2] % 32) != 0:
            p = (x.shape[2] % 64) // 2
            x  = nn.functional.pad(inputs["input"], (p,p,p,p), mode='reflect') 

        # x = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
        x = self.model(x)

        # revert padding
        if p is not None:
            x = x[:,:,p:-p,p:-p]

        x = self.head(x)

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

        builtupmap = torch.sigmoid(x[:,2]) 

        # plot_2dmatrix(popdensemap[0])
        # plot_2dmatrix(inputs["input"][0]*0.2+0.5)

        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap}



class ResBlocksSqueeze(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim):
        super(ResBlocksSqueeze, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        k1a = 3
        k1b = 3
        k2 = 3

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            nn.MaxPool2d(4),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),

            Block(feature_dim, k1=k1a, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            Block(feature_dim, k1=k1b, k2=k2),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            Block(feature_dim),
        )
        self.head = nn.Conv2d(feature_dim, 4, kernel_size=1, padding=0)

        a = torch.zeros((1,3,128,128))
        a[0,:,64,64] = 500
        plot_2dmatrix(self.model(a)[0,0].abs()>0.000001)

        params_sum = sum(p.numel() for p in self.model.parameters() if p.requires_grad)    

                                  
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
            p = (x.shape[2] % 64) // 2
            x  = nn.functional.pad(inputs["input"], (p,p,p,p), mode='reflect') 

        # x = nn.functional.pad(inputs["input"], self.p2d, mode='reflect')
        x = self.model(x)

        # revert padding
        if p is not None:
            x = x[:,:,p:-p,p:-p]

        x = self.head(x)

        # Population map
        popdensemap = nn.functional.softplus(x[:,0])
        popcount = popdensemap.sum((1,2))

        # Building map
        builtdensemap = nn.functional.softplus(x[:,1])
        builtcount = builtdensemap.sum((1,2))

        builtupmap = torch.sigmoid(x[:,2]) 

        # plot_2dmatrix(popdensemap[0])
        # plot_2dmatrix(inputs["input"][0]*0.2+0.5)

        return {"popcount": popcount, "popdensemap": popdensemap,
                "builtdensemap": builtdensemap, "builtcount": builtcount,
                "builtupmap": builtupmap}
    

class UResBlocks(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim):
        super(UResBlocks, self).__init__()

        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        k1a = 3
        k1b = 3
        k2 = 1

        s2 = 2
        self.enc1 = nn.Sequential(
            # nn.Conv2d(input_channels, feature_dim, kernel_size=5, padding=2, stride=2), nn.ReLU(),
            nn.Conv2d(input_channels, feature_dim, kernel_size=7, padding=(7-1)//2, stride=s2), nn.ReLU(),
            Block(feature_dim, k1=k1a, k2=k1b),
            # Block(feature_dim, k1=k1b, k2=k2)
        )

        s1 = 2
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=s1, stride=s1, padding=(s1-1)//2),
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim*2, k1=k1a, k2=k2),
            Block(feature_dim*2, k1=k2, k2=k2),
            nn.Upsample(scale_factor=s1, mode='nearest')
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(feature_dim*(1+2), feature_dim, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim, k1=k1b, k2=k2),
            nn.Upsample(scale_factor=s2, mode='nearest')
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(feature_dim + input_channels, feature_dim//2, kernel_size=3, padding=1), nn.ReLU(),
            Block(feature_dim//2, k1=k1b, k2=k2)
        )

        self.head = nn.Conv2d(feature_dim//2, 4, kernel_size=1, padding=0)

        params_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        a = torch.zeros((1,3,128,128))
        a[0,:,64,64] = 500
        plot_2dmatrix(self.backbone(a)[0,0].abs()>0.00000001)
        # plot_2dmatrix(self.backbone(a)[0,0])

    def backbone(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f1b = self.dec2(torch.cat([f2,f1],1))
        featend = self.dec1(torch.cat([f1b, x],1))
        return featend

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
        
        # forward backbone
        featend = self.backbone(x)

        # revert padding
        if p is not None:
            featend = featend[:,:,p:-p,p:-p]

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
    def __init__(self, input_channels, feature_dim, k1a=3, k1b=3):
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
    
                                  
    def forward(self, inputs, train=False, alpha=0.1):

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
        

    def forward(self, inputs, train=False, alpha=0.1):

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
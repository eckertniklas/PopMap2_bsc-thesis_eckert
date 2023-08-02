
 
import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp
from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, DomainClassifier_v6, ReverseLayerF
from model.customUNet import CustomUNet
from torch.nn.functional import upsample_nearest, interpolate

from utils.plot import plot_2dmatrix, plot_and_save



class POMELO_module(nn.Module):
    '''
    PomeloUNet
    Description:
        - UNet with a regression head

    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", classifier="v1", head="v1", down=5,
                occupancymodel=False, pretrained=False, dilation=1, replace7x7=True):
        super(POMELO_module, self).__init__()
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
        self.occupancymodel = occupancymodel
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        # Build the main model
        self.unetmodel = CustomUNet(feature_extractor, in_channels=input_channels, classes=feature_dim, down=self.down, dilation=dilation, replace7x7=replace7x7)

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

        # lift the bias of the head
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

                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True, encoder_no_grad=False, unet_no_grad=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """

        # Add padding
        data, (px1,px2,py1,py2) = self.add_padding(inputs["input"], padding)

        # Forward the main model
        if  unet_no_grad:
            with torch.no_grad():
                features, decoder_features = self.unetmodel(data, return_features=return_features, encoder_no_grad=encoder_no_grad)
        else:
            features, decoder_features = self.unetmodel(data, return_features=return_features, encoder_no_grad=encoder_no_grad)

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
        popvarmap = nn.functional.softplus(out[:,1])

        if self.occupancymodel:
            popdensemap = nn.functional.softplus(out[:,0]) 
            if "building_counts" in inputs.keys():
                scale = popdensemap.clone().cpu().detach().numpy()
                popdensemap = popdensemap * inputs["input"][0,-1]
                popvarmap = popvarmap * inputs["input"][0,-1].squeeze(1)
            else:
                raise ValueError("building_counts not in inputs.keys()")
        else:
            popdensemap = nn.functional.relu(out[:,0])
            popvarmap = nn.functional.softplus(out[:,1])
        
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            popcount = (popdensemap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popcount = popdensemap.sum((1,2))

        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            popvar = (popvarmap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popvar = popvarmap.sum((1,2))

        # Building map
        # builtdensemap = nn.functional.softplus(out[:,2])
        # builtcount = builtdensemap.sum((1,2))

        # # Builtup mask
        # builtupmap = torch.sigmoid(out[:,3])

        return {"popcount": popcount, "popdensemap": popdensemap,
                "popvar": popvar ,"popvarmap": popvarmap, 
                # "builtdensemap": builtdensemap, "builtcount": builtcount,
                # "builtupmap": builtupmap,
                "scale": scale,
                "domain": domain, "features": features,
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

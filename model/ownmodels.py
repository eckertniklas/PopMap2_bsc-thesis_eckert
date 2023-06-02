


import torch
from torch import nn

import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp
from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, DomainClassifier_v6, ReverseLayerF
from model.customUNet import CustomUNet
from torch.nn.functional import upsample_nearest, interpolate

from utils.plot import plot_2dmatrix, plot_and_save





class BoostUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", classifier="v1", down=2, down2=4):
        super(BoostUNet, self).__init__()
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        # Build the main model
        self.unetmodel1 = CustomUNet(feature_extractor, in_channels=2, classes=feature_dim, down=down) # Sentinel1
        self.unetmodel2 = CustomUNet(feature_extractor, in_channels=input_channels-2+1, classes=feature_dim, down=down2) # Sentinel2

        # Define batchnorm layer for the feature extractor
        # self.bn = nn.BatchNorm2d(input_channels)
        # self.maxi = torch.tensor([0,0,0,0,0,0], dtype=torch.float32)

        # Build the regression head
        self.head1 = nn.Conv2d(feature_dim, 5, kernel_size=1, padding=0)
        self.head2 = nn.Conv2d(feature_dim, 5, kernel_size=1, padding=0)

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
        self.params_sum = sum(p.numel() for p in self.unetmodel1.parameters() if p.requires_grad)
        self.params_sum = sum(p.numel() for p in self.unetmodel2.parameters() if p.requires_grad)

                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True):

        data = inputs["input"]

        # Add padding
        data, (px1,px2,py1,py2) = self.add_padding(inputs["S1"], padding)

        #### STAGE 1 ####
        # Forward the main model1
        # self.maxi[data.max(dim=2)[0].max(dim=2)[0].max(dim=0)[0].cpu()>self.maxi] = data.max(dim=2)[0].max(dim=2)[0].max(dim=0)[0].cpu()[data.max(dim=2)[0].max(dim=2)[0].max(dim=0)[0].cpu()>self.maxi]
        features, decoder_features = self.unetmodel1(data, return_features=return_features)

        # revert padding
        # features = self.revert_padding(features, (px1,px2,py1,py2))

        # Forward the head1
        out = self.head1(features)

        # Foward the domain classifier
        if self.domain_classifier is not None and return_features and alpha>0:
            reverse_features = ReverseLayerF.apply(decoder_features.unsqueeze(3), alpha) # apply gradient reversal layer
            domain_raw = self.domain_classifier(reverse_features.permute(0,2,3,1).reshape(-1, reverse_features.size(1))).view(reverse_features.size(0),-1)
        else:
            domain_raw = None

        # Population map and total count, raw
        popdensemap_raw = nn.functional.relu(out[:,0])
        if "admin_mask" in inputs.keys():
            popcount_raw = (popdensemap_raw * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popcount_raw = popdensemap_raw.sum((1,2))

        popvarmap_raw = nn.functional.softplus(out[:,1])
        if "admin_mask" in inputs.keys():
            popvar_raw = (popvarmap_raw * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popvar_raw = popvarmap_raw.sum((1,2))

        
        #### STAGE 2 ####

        # Add padding
        data, (px1,px2,py1,py2) = self.add_padding(inputs["S2"], padding)

        # Concatenate the output of the main model with the input of the second model
        newinput = torch.cat([data, popdensemap_raw.unsqueeze(1)], dim=1)

        # Forward the main model2
        features, decoder_features = self.unetmodel2(newinput, return_features=return_features)

        # revert padding
        features = self.revert_padding(features, (px1,px2,py1,py2))

        # Forward the head2
        out = self.head2(features)

        # Foward the domain classifier
        if self.domain_classifier is not None and return_features and alpha>0:
            reverse_features = ReverseLayerF.apply(decoder_features.unsqueeze(3), alpha) # apply gradient reversal layer
            domain = self.domain_classifier(reverse_features.permute(0,2,3,1).reshape(-1, reverse_features.size(1))).view(reverse_features.size(0),-1)
        else:
            domain = None
        


        # Population map and total count
        popdensemap = nn.functional.relu(out[:,0])
        if "admin_mask" in inputs.keys():
            popcount = (popdensemap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popcount = popdensemap.sum((1,2))

        popvarmap = nn.functional.softplus(out[:,1])
        if "admin_mask" in inputs.keys():
            popvar = (popvarmap * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
        else:
            popvar = popvarmap.sum((1,2))


        # Building map
        builtdensemap = nn.functional.softplus(out[:,2])
        builtcount = builtdensemap.sum((1,2))

        # Builtup mask
        builtupmap = torch.sigmoid(out[:,3])


        return {"popcount": popcount, "popdensemap": popdensemap, "popvar": popvar ,"popvarmap": popvarmap, 
                "intermediate": {"popcount": popcount_raw, "popdensemap": popdensemap_raw, "popvar": popvar_raw ,"popvarmap": popvarmap_raw, "domain": domain_raw }, 
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

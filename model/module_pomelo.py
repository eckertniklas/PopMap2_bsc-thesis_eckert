
 
import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp
from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, DomainClassifier_v6, ReverseLayerF
from model.customUNet import CustomUNet
from torch.nn.functional import upsample_nearest, interpolate

from model.DDA_model.utils.networks import load_checkpoint

from utils.siren import Siren, Siren1x1

from utils.plot import plot_2dmatrix, plot_and_save

import os
from utils.utils import Namespace


class POMELO_module(nn.Module):
    '''
    PomeloUNet
    Description:
        - UNet with a regression head

    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", down=5,
                occupancymodel=False, pretrained=False, dilation=1, replace7x7=True,
                parent=None, experiment_folder=None, useposembedding=False):
        super(POMELO_module, self).__init__()
        """
        Args:
            - input_channels (int): number of input channels
            - feature_dim (int): number of output channels of the feature extractor
            - feature_extractor (str): name of the feature extractor
            - classifier (str): name of the classifier
            - head (str): name of the regression head
            - down (int): number of downsampling steps in the feature extractor
            - occupancymodel (bool): whether to use the occupancy model
            - pretrained (bool): whether to use the pretrained feature extractor
            - dilation (int): dilation factor
            - replace7x7 (bool): whether to replace the 7x7 convolutions with 3 3x3 convolutions
        """

        self.down = down
        self.occupancymodel = occupancymodel
        self.useposembedding = useposembedding
        self.feature_extractor = feature_extractor
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        # own parent file
        parent_file = os.path.join(experiment_folder, "parent.txt")

        if os.path.exists(parent_file):
            loaded_parent = open(parent_file, "r").readline().strip()
            assert os.path.exists(loaded_parent)
            if parent is not None:
                assert loaded_parent == parent
            parent = loaded_parent

        elif parent is not None:
            # means that this is a new model and we need to create the parent file
            with open(parent_file, "w") as f:
                f.write(parent)
        
        if parent is not None or os.path.exists(parent_file):
            # recursive loading of the boosting model
            self.parent = POMELO_module(input_channels, feature_dim, feature_extractor, down, occupancymodel=occupancymodel,
                                        useposembedding=useposembedding, experiment_folder=parent)
            self.parent.unetmodel.load_state_dict(torch.load(os.path.join(parent, "last_unetmodel.pth"))["model"])
            self.parent.head.load_state_dict(torch.load(os.path.join(parent, "last_head.pth"))["model"])

            if self.useposembedding:
                self.parent.embedder.load_state_dict(torch.load(os.path.join(parent, "last_embedder.pth"))["model"])
        else:
            # create the parent file
            self.parent = None

        this_input_dim = input_channels if self.parent is None else input_channels + 1

        if useposembedding:
            self.embedder = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            )

            # self.embedder = nn.Sequential(
            #     nn.Conv2d(20, 32, kernel_size=1, padding=0), nn.GELU(),
            #     nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.GELU(),
            #     nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.GELU(),
            # )


            # self.embedder = nn.Sequential(
            #     nn.Conv2d(20, 64, kernel_size=1, padding=0), nn.GELU(),
            #     nn.Conv2d(64, 64, kernel_size=1, padding=0), nn.GELU(),
            #     nn.Conv2d(64, 64, kernel_size=1, padding=0), nn.GELU(),
            #     nn.Conv2d(64, feature_dim, kernel_size=1, padding=0), nn.GELU(),
            # )

            # self.embedder = nn.Sequential(
            #     nn.Conv2d(20, 64, kernel_size=1, padding=0), nn.ReLU(),
            #     nn.Conv2d(64, 64, kernel_size=1, padding=0), nn.ReLU(),
            #     nn.C

            # self.embedder = nn.Sequential(
            #     Siren1x1(2, 32, w0=30., is_first=True),
            #     Siren1x1(32, 32, w0=1.),
            #     # Siren1x1(32, feature_dim, w0=1, activation=torch.nn.Identity())
            #     Siren1x1(32, feature_dim, w0=1, activation=torch.nn.Tanh())
            # )
            this_input_dim += feature_dim

        self.head = nn.Conv2d(feature_dim, 5, kernel_size=1, padding=0)

        # Build the main model
        if feature_extractor=="DDA":
                # get model
                MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[64, 128,] )
                CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.5)
                PATHS = Namespace(OUTPUT="/scratch2/metzgern/HAC/data/DDAdata/outputsDDA")
                DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
                TRAINER = Namespace(LR=1e5)
                cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS,
                                DATALOADER=DATALOADER, TRAINER=TRAINER, NAME="fusionda_new")

                ## load weights from checkpoint
                self.unetmodel, _, _ = load_checkpoint(epoch=15, cfg=cfg, device="cuda", no_disc=True)
        else:
            self.unetmodel = CustomUNet(feature_extractor, in_channels=this_input_dim, classes=feature_dim, 
                                        down=self.down, dilation=dilation, replace7x7=replace7x7, pretrained=pretrained)
        # lift the bias of the head
        self.head.bias.data = 0.75 * torch.ones(5)

        # calculate the number of parameters
        self.params_sum = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

                                  
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True, encoder_no_grad=False, unet_no_grad=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """
        aux = {}

        # forward the parent model without gradient if exists
        if self.parent is not None:
            # Forward the parent model
            with torch.no_grad():
                output_dict = self.parent(inputs, padding=False, return_features=False, unet_no_grad=unet_no_grad)
            # Concatenate the parent features with the input
            inputdata = torch.cat([ output_dict["popdensemap"].unsqueeze(1), inputs["input"]], dim=1)
        else:
            inputdata = inputs["input"]

        # Embed the pose information
        if self.useposembedding:
            if isinstance(self.embedder[0], Siren1x1):
                xy = torch.cat([  inputs["positional_encoding"][:,0].unsqueeze(0),
                                  inputs["positional_encoding"][:,inputs["positional_encoding"].shape[1]//2].unsqueeze(0) ],
                                  dim=1)
                pose = self.embedder(xy)
            else:
                pose = self.embedder(inputs["positional_encoding"])

            # Concatenate the pose embedding to the input data
            inputdata = torch.cat([inputdata, pose], dim=1)

        else:
            inputdata = inputdata


        # Add padding
        data, (px1,px2,py1,py2) = self.add_padding(inputdata, padding)

        # Forward the main model
        if self.feature_extractor=="DDA":
            x_fusion = torch.cat([data[:, 4:6], # S1
                                  torch.flip(data[:, :3],dims=(1,)), # S2_RGB
                                  data[:, 3:4]], # S2_NIR
                                  dim=1)
            
            _, _, fusion_logits, _, _ = self.unetmodel(x_fusion, alpha=0, encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad)

            # repeat along dim 1
            out = fusion_logits.repeat(1, 2, 1, 1)
            out = self.revert_padding(out, (px1,px2,py1,py2))

        else:

            if unet_no_grad:
                with torch.no_grad():
                    features, decoder_features = self.unetmodel(data, return_features=return_features, encoder_no_grad=encoder_no_grad)
            else:
                features, decoder_features = self.unetmodel(data, return_features=return_features, encoder_no_grad=encoder_no_grad)

            # revert padding
            features = self.revert_padding(features, (px1,px2,py1,py2))

            aux["features"] = features
            aux["decoder_features"] = decoder_features

            # Forward the head
            # out_raw = self.head(features)
            out = self.head(features)

        # Population map and total count
        # popvarmap_raw = nn.functional.softplus(out_raw[:,1])
        popvarmap = nn.functional.softplus(out[:,1])

        # popdensemap_raw = nn.functional.relu(out_raw[:,0])
        popdensemap = nn.functional.relu(out[:,0])

        # popdensemap = (popdensemap*1.8 + popdensemap_raw*0.2) / 2
        # popdensemap = (popdensemap + popdensemap_raw) / 2

        if self.occupancymodel:
            # for raw
            if "building_counts" in inputs.keys():
                # aux["scale_raw"] = popdensemap_raw.clone().cpu().detach()
                # popdensemap_raw = popdensemap_raw * inputs["building_counts"][:,0]
                # popvarmap_raw = popvarmap_raw * inputs["building_counts"][:,0]
                # for final
                aux["scale"] = popdensemap.clone().cpu().detach()
                popdensemap = popdensemap * inputs["building_counts"][:,0]
                popvarmap = popvarmap * inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys()")
        else:
            # popdensemap_raw = nn.functional.relu(out_raw[:,0])
            # popvarmap_raw = nn.functional.softplus(out_raw[:,1])
            popdensemap = nn.functional.relu(out[:,0])
            popvarmap = nn.functional.softplus(out[:,1])
        
        # aggregate the population counts
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            this_mask = inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1)
            # popcount_raw = (popdensemap_raw * this_mask).sum((1,2))
            popcount = (popdensemap * this_mask).sum((1,2))
            # popvar_raw = (popvarmap_raw * this_mask).sum((1,2))
            popvar = (popvarmap * this_mask).sum((1,2))
        else:
            # popcount_raw = popdensemap_raw.sum((1,2))
            popcount = popdensemap.sum((1,2))
            # popvar_raw = popvarmap_raw.sum((1,2))
            popvar = popvarmap.sum((1,2))


        return {"popcount": popcount, "popdensemap": popdensemap,
                "popvar": popvar ,"popvarmap": popvarmap, 
                # "builtdensemap": builtdensemap, "builtcount": builtcount,
                # "builtupmap": builtupmap,
                # "intermediate": {"popcount": popcount_raw, "popdensemap": popdensemap_raw, "popvar": popvar_raw,
                # "popvarmap": popvarmap_raw, "domain": None, "decoder_features": None}, 
                **aux,
                # "features": features,
                # "decoder_features": decoder_features
                }


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


 
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
                parent=None, experiment_folder=None, useposembedding=False, head="v1"):
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
        self.head_name = head
        head_input_dim = 0
        
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
                                        useposembedding=useposembedding, experiment_folder=parent, replace7x7=replace7x7, head=head)
            self.parent.unetmodel.load_state_dict(torch.load(os.path.join(parent, "last_unetmodel.pth"))["model"])
            self.parent.head.load_state_dict(torch.load(os.path.join(parent, "last_head.pth"))["model"])

            if self.useposembedding:
                self.parent.embedder.load_state_dict(torch.load(os.path.join(parent, "last_embedder.pth"))["model"])
        else:
            # create the parent file
            self.parent = None

        # this_input_dim = input_channels if self.parent is None else input_channels + 1
        # this_input_dim = input_channels if self.parent is None else input_channels + feature_dim #old
        head_input_dim = head_input_dim if self.parent is None else head_input_dim + feature_dim
        this_input_dim = input_channels


        if useposembedding:
            freq = 4 # for 2 dimensions and 2 components (sin and cos)
            self.embedder = nn.Sequential(
                nn.Conv2d(2*freq, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            )

            head_input_dim += feature_dim


        if head=="v1":
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, 2, kernel_size=1, padding=0)
            )
            this_input_dim += feature_dim

        elif head=="v2":
            h = 64
            head_input_dim += feature_dim
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )

        elif head=="v3":
            h = 64
            head_input_dim += feature_dim + 1
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
            this_input_dim -= 1 # no building footprint

        elif head=="v4":
            #footprint at the front and middle input
            h = 64
            head_input_dim += feature_dim + 1
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
        elif head=="v5":
            freq = 4 # for 2 dimensions and 2 components (sin and cos)
            self.embedder = nn.Sequential(
                nn.Conv2d(2*freq, 32, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.ReLU(),
            )
            h = 64
            head_input_dim += feature_dim + 1
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
            this_input_dim -= 1 # no building footprint




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
            
        # lift the bias of the head to avoid the risk of dying ReLU
        self.head[-1].bias.data = 0.75 * torch.ones(2)

        # calculate the number of parameters
        self.params_sum = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

        # print size of the embedder and head network
        if hasattr(self, "embedder"):
            print("Embedder: ",sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)) 
        print("Head: ",sum(p.numel() for p in self.head.parameters() if p.requires_grad))

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
            parent_features = output_dict["features"]
            inputdata = torch.cat([ output_dict["features"], inputs["input"]], dim=1) #old
            # inputdata = torch.cat([ output_dict["popdensemap"].unsqueeze(1), inputs["input"]], dim=1)
        else:
            inputdata = inputs["input"]

        # Embed the pose information
        if self.useposembedding:
            if isinstance(self.embedder[0], Siren1x1):
                # only use the first sine and cosine frequency
                xy = torch.cat([  inputs["positional_encoding"][:,0].unsqueeze(0),
                                  inputs["positional_encoding"][:,inputs["positional_encoding"].shape[1]//2].unsqueeze(0) ],
                                  dim=1)
                pose = self.embedder(xy)
            else:
                # TODO: optimize for occupancy model
                if self.occupancymodel:
                    pose = self.sparse_forward(inputs["positional_encoding"], inputs["building_counts"][:,0]>0, self.embedder, out_channels=8)
                # pose = self.embedder(inputs["positional_encoding"])

            # Concatenate the pose embedding to the input data

            if self.head_name in ["v2", "v4"]:
                inputdata = inputdata
            elif self.head_name in ["v3"]:
                inputdata = inputdata[:,0:-1] # remove the building footprint from the variables
            else:
                inputdata = torch.cat([inputdata, pose], dim=1)

        else:
            if self.head_name in ["v2", "v4"]:
                inputdata = inputdata
            elif self.head_name in ["v3"]:
                inputdata = inputdata[:,0:-1] # remove the building footprint from the variables
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

            if self.head_name in ["v2"]:
                out = self.head(torch.cat([features, pose], dim=1))
            elif self.head_name in ["v3", "v4"]:
                if self.parent is not None:
                    out = self.head(torch.cat([features, pose, output_dict["popdensemap"], inputs["building_counts"]], dim=1))
                else:
                    if self.occupancymodel:
                        # TODO: switch to sparse convolutions, since self.head is an MLP, and the output is eventually sparse anyways
                        
                        # prepare the input to the head
                        if self.useposembedding:
                            headin = torch.cat([features, pose, inputs["building_counts"]], dim=1)
                        else:
                            headin = torch.cat([features, inputs["building_counts"]], dim=1)

                        out = self.sparse_forward(headin, inputs["building_counts"][:,0]>0, self.head, out_channels=8)


                        # # bring everything together
                        # batch_size, channels, height, width = headin.shape
                        # try:
                        #     headin_flat = headin.permute(1,0,2,3).view(channels, -1, 1)
                        # except:
                        #     headin_flat = headin.permute(1,0,2,3).reshape(channels, -1, 1)
                        # mask = inputs["building_counts"][:,0]>0
                        # mask_flat = mask.view(-1)
                        # headin_flat_masked = headin_flat[:, mask_flat]

                        # # flatten

                        # # initialize the output
                        # out_flat = torch.zeros((2, batch_size*height*width,1), device=headin.device)
                        
                        # # perform the forward pass
                        # out_flat[ :, mask_flat] = self.head(headin_flat_masked)
                        
                        # # reshape the output
                        # out = out_flat.view(2, batch_size, height, width).permute(1,0,2,3)


                        # out2 = self.head(torch.cat([features, inputs["building_counts"]], dim=1))

                    else:
                        if self.useposembedding:
                            out = self.head(torch.cat([features, pose, inputs["building_counts"]], dim=1))
                        else:
                            out = self.head(torch.cat([features, inputs["building_counts"]], dim=1))
            else:
                out = self.head(features)

        # Population map and total count
        # popvarmap_raw = nn.functional.softplus(out_raw[:,1])
        # popvarmap = nn.functional.softplus(out[:,1])

        # popdensemap_raw = nn.functional.relu(out_raw[:,0])
        # popdensemap = nn.functional.relu(out[:,0])

        # popdensemap = (popdensemap*1.8 + popdensemap_raw*0.2) / 2
        # popdensemap = (popdensemap + popdensemap_raw) / 2

        if self.occupancymodel:

            # activation function
            popvarmap = nn.functional.softplus(out[:,1])
            # popdensemap = nn.functional.softplus(out[:,0])
            popdensemap = nn.functional.relu(out[:,0])

            # for raw
            if "building_counts" in inputs.keys():
                # aux["scale_raw"] = popdensemap_raw.clone().cpu().detach()
                # popdensemap_raw = popdensemap_raw * inputs["building_counts"][:,0]
                # popvarmap_raw = popvarmap_raw * inputs["building_counts"][:,0]
                # for final
                # aux["scale"] = popdensemap.clone().cpu().detach()
                aux["scale"] = popdensemap.clone()
                # aux["scale"][inputs["building_counts"][:,0]==0] = 0

                popdensemap = popdensemap * inputs["building_counts"][:,0]
                # popdensemap = popdensemap * (inputs["building_counts"][:,0]>0.25)
                # popdensemap = popdensemap * (inputs["building_counts"][:,0]>0.25) * inputs["building_counts"][:,0]
                # popvarmap = popvarmap #* inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys()")
        else:
            # popdensemap_raw = nn.functional.relu(out_raw[:,0])
            # popvarmap_raw = nn.functional.softplus(out_raw[:,1])
            popdensemap = nn.functional.relu(out[:,0])
            popvarmap = nn.functional.softplus(out[:,1])
            aux["scale"] = popdensemap.clone().cpu().detach()
            aux["scale"] = None
        
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


    def sparse_forward(self, inp, mask, module, out_channels=2):

        # bring everything together
        batch_size, channels, height, width = inp.shape
        try:
            inp_flat = inp.permute(1,0,2,3).view(channels, -1, 1)
        except:
            inp_flat = inp.permute(1,0,2,3).reshape(channels, -1, 1)

        # flatten mask
        mask_flat = mask.view(-1)
        
        # apply mask to the input
        inp_flat_masked = inp_flat[:, mask_flat]

        # initialize the output
        out_flat = torch.zeros((out_channels, batch_size*height*width,1), device=inp.device)
        
        # perform the forward pass
        out_flat[ :, mask_flat] = module(inp_flat_masked)
        
        # reshape the output
        out = out_flat.view(out_channels, batch_size, height, width).permute(1,0,2,3)

        return out
    

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

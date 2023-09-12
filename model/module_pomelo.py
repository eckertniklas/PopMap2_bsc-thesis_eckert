
 
import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp
# from model.DANN import DomainClassifier, DomainClassifier1x1, DomainClassifier_v3, DomainClassifier_v4, DomainClassifier_v5, DomainClassifier_v6, ReverseLayerF
from model.customUNet import CustomUNet
from torch.nn.functional import upsample_nearest, interpolate
import ast

from model.DDA_model.utils.networks import load_checkpoint

from utils.siren import Siren, Siren1x1

from utils.plot import plot_2dmatrix, plot_and_save

import os
from utils.utils import Namespace

from utils.utils import read_params

class POMELO_module(nn.Module):
    '''
    PomeloUNet
    Description:
        - UNet with a regression head

    '''
    def __init__(self, input_channels, feature_dim, feature_extractor="resnet18", down=5,
                occupancymodel=False, pretrained=False, dilation=1, replace7x7=True,
                parent=None, experiment_folder=None, useposembedding=False, head="v1", grouped=False,
                lempty_eps=0.0, dropout=0.0):
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
            - parent (str): path to the parent model
            - experiment_folder (str): path to the experiment folder
            - useposembedding (bool): whether to use the pose embedding
            - grouped (bool): whether to use grouped convolutions

        """

        self.down = down
        self.occupancymodel = occupancymodel
        self.useposembedding = useposembedding
        self.feature_extractor = feature_extractor
        self.head_name = head
        head_input_dim = 0
        this_input_dim = input_channels
        head_input_dim = head_input_dim
        if lempty_eps>0:
            self.lempty_eps = torch.nn.Parameter(torch.tensor(lempty_eps), requires_grad=True)
        else:
            self.lempty_eps = 0.0
        
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

            # load csv to dict
            argsdict = read_params(os.path.join(parent, "args.csv"))

            print("-"*50)
            print("Loading parent model from: ", parent)
            self.parent = POMELO_module(input_channels, int(argsdict["feature_dim"]), argsdict["feature_extractor"], int(argsdict["down"]), occupancymodel=ast.literal_eval(argsdict["occupancymodel"]),
                                        useposembedding=ast.literal_eval(argsdict["useposembedding"]), experiment_folder=parent, replace7x7=ast.literal_eval(argsdict["replace7x7"]), head=str(argsdict["head"]),
                                        grouped=ast.literal_eval(argsdict["grouped"]))
            self.parent.unetmodel.load_state_dict(torch.load(os.path.join(parent, "last_unetmodel.pth"))["model"])
            self.parent.head.load_state_dict(torch.load(os.path.join(parent, "last_head.pth"))["model"])

            if ast.literal_eval(argsdict["useposembedding"]):
                self.parent.embedder.load_state_dict(torch.load(os.path.join(parent, "last_embedder.pth"))["model"])

            head_input_dim += 1
        else:
            # create the parent file
            self.parent = None

        if useposembedding:
            if head=="v3":
                freq = 2 # for x dimensions and 2 components (sin and cos)
                self.embedder = nn.Sequential(
                    nn.Conv2d(2*freq, 32, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                    nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                )
                self.embedding_dim = feature_dim
                head_input_dim += feature_dim
            elif head=="v4":
                freq = 2 # for x dimensions and 2 components (sin and cos)
                self.embedder = nn.Sequential(
                    nn.Conv2d(2*freq, 32, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                    nn.Conv2d(32, feature_dim, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                )
                self.embedding_dim = feature_dim
                head_input_dim += feature_dim

        if head=="v3":
            h = 64
            head_input_dim += feature_dim + 1
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
        
        elif head=="v4":
            h = 64
            head_input_dim += feature_dim + 1
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True), 
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )

        elif head=="v6":
            h = 128
            head_input_dim += feature_dim + 1
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )


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
            if this_input_dim>0:
                self.unetmodel = CustomUNet(feature_extractor, in_channels=this_input_dim, classes=feature_dim, 
                                            down=self.down, dilation=dilation, replace7x7=replace7x7, pretrained=pretrained,
                                            grouped=grouped, dropout=dropout)
            else:
                self.unetmodel = None

        # lift the bias of the head to avoid the risk of dying ReLU
        self.head[-1].bias.data = 0.75 * torch.ones(2)

        # print size of the embedder and head network
        self.num_params = 0
        if hasattr(self, "embedder"):
            print("Embedder: ",sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)) 
            self.num_params += sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)
        print("Head: ",sum(p.numel() for p in self.head.parameters() if p.requires_grad))
        self.num_params += sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        self.num_params += self.unetmodel.num_params if self.unetmodel is not None else 0



    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True,
                encoder_no_grad=False, unet_no_grad=False, sparse=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """

        X = inputs["input"]

        if self.lempty_eps>0:
            inputs["building_counts"][:,0] = inputs["building_counts"][:,0] + self.lempty_eps

        if sparse:
            # create sparsity mask
            sub = 60
            sparsity_mask = (inputs["building_counts"][:,0]>0) * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
            xindices = torch.ones(sparsity_mask.shape[1]).multinomial(num_samples=min(sub,sparsity_mask.shape[1]), replacement=False).sort()[0]
            yindices = torch.ones(sparsity_mask.shape[2]).multinomial(num_samples=min(sub,sparsity_mask.shape[2]), replacement=False).sort()[0]
            sparsity_mask[:, xindices.unsqueeze(1), yindices] = 1

        aux = {}

        # forward the parent model without gradient if exists
        middlefeatures = []
        if self.parent is not None:
            # Forward the parent model
            with torch.no_grad():
                output_dict = self.parent(inputs, padding=False, return_features=False, unet_no_grad=unet_no_grad, sparse=sparse)

            # Concatenate the parent features with middle features of the current model
            middlefeatures.append(output_dict["scale"].unsqueeze(1))
            
        # Embed the pose information
        if self.useposembedding:
        
            # optimized for occupancy model
            if self.occupancymodel:
                if sparse: 

                    # downsample the feature map
                    lazy_pos = True
                    if lazy_pos:
                        pose = F.interpolate(inputs["positional_encoding"], size=(20, 20), mode='bilinear', align_corners=False)
                        pose = self.embedder(pose)
                        pose = F.interpolate(pose, size=(inputs["positional_encoding"].shape[2], inputs["positional_encoding"].shape[3]), mode='bilinear', align_corners=False)
                    else:
                        pose = self.sparse_forward(inputs["positional_encoding"], sparsity_mask, self.embedder, out_channels=self.embedding_dim)

                else:
                    pose = self.embedder(inputs["positional_encoding"])

            # Concatenate the pose embedding to the input data
            if self.head_name in ["v3", "v4", "v6"]:
                X = X 
            else:
                X = torch.cat([X, pose], dim=1)

        else:
            if self.head_name in ["v3", "v4","v6"]: 
                X = X
            else:
                X = X
 
        # Forward the main model
        if self.feature_extractor=="DDA":
            X, (px1,px2,py1,py2) = self.add_padding(X, padding)
            X = torch.cat([X[:, 4:6], # S1
                                  torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                                  X[:, 3:4]], # S2_NIR
                                  dim=1)
            
            _, _, X, _, _ = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad)

            # repeat along dim 1
            out = X.repeat(1, 2, 1, 1)
            out = self.revert_padding(out, (px1,px2,py1,py2))

        else:
            if self.unetmodel is not None: 
                X, (px1,px2,py1,py2) = self.add_padding(X, padding)
                if unet_no_grad:
                    with torch.no_grad():
                        features, _ = self.unetmodel(X, return_features=return_features, encoder_no_grad=encoder_no_grad)
                else:
                    sparse_unet = False
                    if sparse_unet:
                        features = self.unetmodel.sparse_forward(X,  return_features=False, encoder_no_grad=encoder_no_grad, sparsity_mask=sparsity_mask)
                    else:
                        features, _ = self.unetmodel(X, return_features=return_features, encoder_no_grad=encoder_no_grad)

                # revert padding
                features = self.revert_padding(features, (px1,px2,py1,py2))
                middlefeatures.append(features)

                # aux["features"] = features
                # aux["decoder_features"] = decoder_features

            # Forward the head
            if self.head_name in ["v3", "v4", "v6"]:

                # append building counts to the middle features
                middlefeatures.append(inputs["building_counts"])

                if self.occupancymodel:
                    
                    # prepare the input to the head
                    if self.useposembedding: 
                        middlefeatures.append(pose)

                    headin = torch.cat(middlefeatures, dim=1)
                    
                    # perform sparse sampling and memory efficient forward pass
                    if sparse:
                        out = self.sparse_forward(headin, sparsity_mask, self.head, out_channels=2)
                    else:
                        out = self.head(headin)

                else:
                    if self.useposembedding:
                        middlefeatures.append(pose)
                    headin = torch.cat(middlefeatures, dim=1)
                    out = self.head(headin)
            else:
                out = self.head(features)

        # Population map and total count
        if self.occupancymodel:

            # activation function
            # popvarmap = nn.functional.softplus(out[:,1])
            scale = nn.functional.relu(out[:,0])

            # for raw
            if "building_counts" in inputs.keys(): 
                
                # save the scale
                if sparse:
                    aux["scale"] = scale[sparsity_mask]
                    aux["empty_scale"] = (scale * (1-inputs["building_counts"][:,0]))[sparsity_mask]
                else:
                    aux["scale"] = scale
                    aux["empty_scale"] = scale * (1-inputs["building_counts"][:,0])

                # Get the population density map
                popdensemap = scale * inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys()")
        else:
            # popdensemap_raw = nn.functional.relu(out_raw[:,0])
            # popvarmap_raw = nn.functional.softplus(out_raw[:,1])
            popdensemap = nn.functional.relu(out[:,0])
            # popvarmap = nn.functional.softplus(out[:,1])
            # aux["scale"] = popdensemap.clone().cpu().detach()
            aux["scale"] = None
        

        # aggregate the population counts
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            this_mask = inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1)
            # popcount_raw = (popdensemap_raw * this_mask).sum((1,2))
            # if popdensemap.dtype==torch.float16:
            #     popcount = (popdensemap * this_mask).sum((1,2)).half()
            # else:
            #     popcount = (popdensemap * this_mask).sum((1,2))

            popcount = (popdensemap * this_mask).sum((1,2))
            
            # popvar_raw = (popvarmap_raw * this_mask).sum((1,2))
            # popvar = (popvarmap * this_mask).sum((1,2))
        else:
            # popcount_raw = popdensemap_raw.sum((1,2))
            popcount = popdensemap.sum((1,2))
            # popvar_raw = popvarmap_raw.sum((1,2))
            # popvar = popvarmap.sum((1,2))


        return {"popcount": popcount, "popdensemap": popdensemap,
                # "popvar": popvar ,"popvarmap": popvarmap, 
                # "builtdensemap": builtdensemap, "builtcount": builtcount,
                # "builtupmap": builtupmap,
                # "intermediate": {"popcount": popcount_raw, "popdensemap": popdensemap_raw, "popvar": popvar_raw,
                # "popvarmap": popvarmap_raw, "domain": None, "decoder_features": None}, 
                **aux,
                }


    def sparse_forward(self, inp: torch.Tensor, mask: torch.Tensor,
                       module: callable, out_channels=2) -> torch.Tensor:
        """
        Description:
            - Perform a forward pass with a module on a sparse input
        Input:
            - inp (torch.Tensor): input data
            - mask (torch.Tensor): mask of the input data
            - module (torch.nn.Module): module to apply
            - out_channels (int): number of output channels
        Output:
            - out (torch.Tensor): output data
        """

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

        
        # perform the forward pass with the module
        a = module(inp_flat_masked)

        # initialize the output
        # out_flat = torch.zeros((out_channels, batch_size*height*width,1), device=inp.device, dtype=inp.dtype)
        out_flat = torch.zeros((out_channels, batch_size*height*width,1), device=a.device, dtype=a.dtype)

        # form together
        # out_flat[ :, mask_flat] = module(inp_flat_masked)
        out_flat[ :, mask_flat] = a
        
        # reshape the output
        out = out_flat.view(out_channels, batch_size, height, width).permute(1,0,2,3)

        return out
    

    def add_padding(self, data: torch.Tensor, force=True) -> torch.Tensor:
        """
        Description:
            - Add padding to the input data
        Input:
            - data (torch.Tensor): input data
            - force (bool): whether to force the padding
        Output:
            - data (torch.Tensor): padded data
        """
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
    
    def revert_padding(self, data: torch.tensor, padding: tuple) -> torch.Tensor:
        """
        Description:
            - Revert the padding of the input data
        Input:
            - data (torch.Tensor): input data
            - padding (tuple): padding parameters
        Output:
            - data (torch.Tensor): padded data
        """
        px1,px2,py1,py2 = padding
        if px1 is not None or px2 is not None:
            data = data[:,:,px1:-px2,:]
        if py1 is not None or py2 is not None:
            data = data[:,:,:,py1:-py2]
        return data

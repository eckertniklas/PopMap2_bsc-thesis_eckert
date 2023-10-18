 
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
                lempty_eps=0.0, dropout=0.0, sparse_unet=False, buildinginput=True, biasinit=0.75,
                sentinelbuildings=True, dda_dir="model/DDA_model/checkpoints/"):
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
        self.buildinginput = buildinginput
        self.feature_extractor = feature_extractor
        self.sparse_unet = sparse_unet
        self.sentinelbuildings = sentinelbuildings

        self.head_name = head 
        head_input_dim = 0

        this_input_dim = input_channels
        
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
        
        self.S1, self.S2 = True, True
        if input_channels==0:
            self.S1, self.S2 = False, False
        elif input_channels==2:
            self.S1, self.S2 = True, False
        elif input_channels==4:
            self.S1, self.S2 = False, True
        
        # Build the main model
        if not self.S1 and not self.S2:
            self.unetmodel = None
            unet_out = 0
        elif feature_extractor=="DDA":
            # get pretrained building extractor model from checkpoint
            stage1feats = 8
            stage2feats = 16
            MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[stage1feats, stage2feats,] )
            CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.5)
            # PATHS = Namespace(OUTPUT="/scratch2/metzgern/HAC/data/DDAdata/outputsDDA")
            PATHS = Namespace(OUTPUT=dda_dir)
            DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
            TRAINER = Namespace(LR=1e5)
            cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS,
                            # DATALOADER=DATALOADER, TRAINER=TRAINER, NAME="fusionda_new")
                            DATALOADER=DATALOADER, TRAINER=TRAINER, NAME=f"fusionda_newAug{stage1feats}_{stage2feats}")

            ## load weights from checkpoint
            # self.unetmodel, _, _ = load_checkpoint(epoch=15, cfg=cfg, device="cuda", no_disc=True)
            self.unetmodel, _, _ = load_checkpoint(epoch=30, cfg=cfg, device="cuda", no_disc=True)

            if not pretrained:
                # initialize weights randomly
                for m in self.unetmodel.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                        
            # unet_out = 8*2
            unet_out = self.S1*stage1feats + self.S2*stage1feats
            num_params_sar = sum(p.numel() for p in self.unetmodel.sar_stream.parameters() if p.requires_grad)
            print("trainable DDA SAR: ", num_params_sar)

            num_params_opt = sum(p.numel() for p in self.unetmodel.optical_stream.parameters() if p.requires_grad)
            print("trainable DDA OPT: ", num_params_opt)

            self.unetmodel.disc = None
            self.unetmodel.num_params = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)
        else:
            if this_input_dim>0:
                self.unetmodel = CustomUNet(feature_extractor, in_channels=this_input_dim, classes=feature_dim, 
                                            down=self.down, dilation=dilation, replace7x7=replace7x7, pretrained=pretrained,
                                            grouped=grouped, dropout=dropout)
                unet_out = feature_dim
            else:
                self.unetmodel = None


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

        if buildinginput:
            head_input_dim += 1

        if head=="v3":
            h = 64
            head_input_dim += unet_out
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
        
        elif head=="v4":
            h = 64
            head_input_dim += unet_out
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True), 
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )

        elif head=="v6":
            h = 128
            head_input_dim += unet_out
            head_input_dim -= feature_dim if this_input_dim==0 else 0
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )

        # lift the bias of the head to avoid the risk of dying ReLU
        self.head[-1].bias.data = biasinit * torch.ones(2)

        # print size of the embedder and head network
        self.num_params = 0
        if hasattr(self, "embedder"):
            print("Embedder: ",sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)) 
            self.num_params += sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)
        print("Head: ",sum(p.numel() for p in self.head.parameters() if p.requires_grad))
        self.num_params += sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        self.num_params += self.unetmodel.num_params if self.unetmodel is not None else 0

        # define urban extractor
        self.define_urban_extractor()



# NEW FORWARD
    def forward(self, inputs, train=False, padding=True, alpha=0.1, return_features=True,
                encoder_no_grad=False, unet_no_grad=False, sparse=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """

        X = inputs["input"]

        # create building score, if not available in the dataset
        if "building_counts" not in inputs.keys() or self.sentinelbuildings:
            with torch.no_grad():
                inputs["building_counts"]  = self.create_building_score(inputs)
            torch.cuda.empty_cache()

        if self.lempty_eps>0:
            inputs["building_counts"][:,0] = inputs["building_counts"][:,0] + self.lempty_eps

        if sparse:
            # create sparsity mask 

            if self.sparse_unet:
                # building_sparsity_mask = (inputs["building_counts"][:,0]>0.0001) 
                building_sparsity_mask = (inputs["building_counts"][:,0]>0.001000) 
                sub = 250
                xindices = torch.ones(building_sparsity_mask.shape[1]).multinomial(num_samples=min(sub,building_sparsity_mask.shape[1]), replacement=False).sort()[0]
                yindices = torch.ones(building_sparsity_mask.shape[2]).multinomial(num_samples=min(sub,building_sparsity_mask.shape[2]), replacement=False).sort()[0]
                subsample_mask = torch.zeros_like(building_sparsity_mask)
                subsample_mask[:, xindices.unsqueeze(1), yindices] = 1
 
                subsample_mask = subsample_mask * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
                subsample_mask_empty = subsample_mask * ~building_sparsity_mask
                mask_empty = (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1)) * ~building_sparsity_mask

                # calculate undersampling ratio
                ratio = mask_empty.sum((1,2)) / ( subsample_mask_empty.sum((1,2)) + 1e-5)
                del mask_empty, subsample_mask

                sparsity_mask = building_sparsity_mask.clone()
                sparsity_mask[:, xindices.unsqueeze(1), yindices] = 1
                
                # clip mask to the administrative region
                sparsity_mask *= (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
                # sparsity_mask = (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
            else:
                if self.occupancymodel:
                    sparsity_mask = (inputs["building_counts"][:,0]>0) * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
                else:
                    sparsity_mask = (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))
                sub = 60
                xindices = torch.ones(sparsity_mask.shape[1]).multinomial(num_samples=min(sub,sparsity_mask.shape[1]), replacement=False).sort()[0]
                yindices = torch.ones(sparsity_mask.shape[2]).multinomial(num_samples=min(sub,sparsity_mask.shape[2]), replacement=False).sort()[0]
                sparsity_mask[:, xindices.unsqueeze(1), yindices] = 1

                # clip mask to the administrative region
                sparsity_mask *= (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))

                if sparsity_mask.sum()==0:
                    sparsity_mask = (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))

        
        aux = {}

        # forward the parent model without gradient if exists
        middlefeatures = []

        if self.buildinginput:
            # append building counts to the middle features
            # TODO: make option to create the building maps from the checkpoint, but not at training time....
            middlefeatures.append(inputs["building_counts"])

        if self.parent is not None:
            # Forward the parent model
            with torch.no_grad():
                output_dict = self.parent(inputs, padding=False, return_features=False, unet_no_grad=unet_no_grad, sparse=sparse)

            # Concatenate the parent features with middle features of the current model
            middlefeatures.append(output_dict["scale"].unsqueeze(1))
            

        # Forward the main model
        if self.unetmodel is not None: 
            X, (px1,px2,py1,py2) = self.add_padding(X, padding)
            if self.feature_extractor=="DDA":
                self.unetmodel.freeze_bn_layers()
                if self.S1 and self.S2:
                    X = torch.cat([
                        X[:, 4:6], # S1
                        torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                        X[:, 3:4]], # S2_NIR
                    dim=1)
                elif self.S1 and not self.S2:
                    X = torch.cat([
                        X, # S1
                        torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
                    dim=1)
                elif not self.S1 and self.S2:
                    X = torch.cat([
                        torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                        torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                        X[:, 3:4]], # S2_NIR
                    dim=1)
                
                # unet_no_grad = True
                # encoder_no_grad = True
                if unet_no_grad:
                    with torch.no_grad():
                        self.unetmodel.eval()
                        if self.sparse_unet and sparse:
                            X = self.unetmodel.sparse_forward(X, sparsity_mask, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)
                            # features = self.unetmodel.sparse_forward(X, sparsity_mask, alpha=0, encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad, return_features=True)
                        else:
                            X = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)
                else:
                    if self.sparse_unet and sparse:
                        X = self.unetmodel.sparse_forward(X, sparsity_mask, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)
                    else:
                        X = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)

            else:
                if unet_no_grad:
                    with torch.no_grad():
                        if self.sparse_unet and sparse:
                            X = self.unetmodel.sparse_forward(X,  return_features=False, encoder_no_grad=encoder_no_grad, sparsity_mask=sparsity_mask)
                        else:
                            X, _ = self.unetmodel(X, return_features=return_features, encoder_no_grad=encoder_no_grad)
                else:
                    if self.sparse_unet and sparse:
                        X = self.unetmodel.sparse_forward(X,  return_features=False, encoder_no_grad=encoder_no_grad, sparsity_mask=sparsity_mask)
                    else:
                        X, _ = self.unetmodel(X, return_features=return_features, encoder_no_grad=encoder_no_grad)

            # revert padding
            X = self.revert_padding(X, (px1,px2,py1,py2))
            middlefeatures.append(X)

        # Embed the pose information
        if self.useposembedding:
        
            lazy_pos = True
            if sparse: 
                # downsample the feature map, and only forward the few anchor points
                if lazy_pos:
                    pose = F.interpolate(inputs["positional_encoding"], size=(20, 20), mode='bilinear', align_corners=False)
                    pose = self.embedder(pose)
                    pose = F.interpolate(pose, size=(inputs["positional_encoding"].shape[2], inputs["positional_encoding"].shape[3]), mode='bilinear', align_corners=False)
                else:
                    pose = self.sparse_module_forward(inputs["positional_encoding"], sparsity_mask, self.embedder, out_channels=self.embedding_dim)

            else:
                pose = self.embedder(inputs["positional_encoding"])

            middlefeatures.append(pose)

        headin = torch.cat(middlefeatures, dim=1)

        # forward the head
        if sparse:
            out = self.sparse_module_forward(headin, sparsity_mask, self.head, out_channels=2)[:,0]
        else:
            out = self.head(headin)[:,0]

        # Population map and total count
        if self.occupancymodel:

            # activation function for the population map is a ReLU to avoid negative values
            scale = nn.functional.relu(out)

            if "building_counts" in inputs.keys(): 
                
                # save the scale
                if sparse and self.sparse_unet:
                    aux["scale"] = torch.cat( [(scale* ratio.view(ratio.shape[0],1,1))[subsample_mask_empty] ,
                                                   scale[building_sparsity_mask] ]  , dim=0)                        
                    aux["empty_scale"] = None
                elif sparse:
                    aux["scale"] = scale[sparsity_mask]
                    aux["empty_scale"] = (scale * (1-inputs["building_counts"][:,0]))[sparsity_mask]
                else:
                    aux["scale"] = scale
                    aux["empty_scale"] = scale * (1-inputs["building_counts"][:,0])

                # Get the population density map
                popdensemap = scale * inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys(), but occupancy model is True")
        else:
            popdensemap = nn.functional.relu(out)
            aux["scale"] = None
            aux["empty_scale"] = None
        
        # aggregate the population counts over the administrative region
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D
            if self.sparse_unet and sparse:
                empty_popcount = (popdensemap * subsample_mask_empty).sum((1,2))  * ratio
                builtup_count = (popdensemap * building_sparsity_mask * (inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1))).sum((1,2))
                popcount = empty_popcount + builtup_count
            else:
                this_mask = inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1)
                popcount = (popdensemap * this_mask).sum((1,2))
            
        else:
            popcount = popdensemap.sum((1,2))

        return {"popcount": popcount, "popdensemap": popdensemap,
                **aux,
                }


    def sparse_module_forward(self, inp: torch.Tensor, mask: torch.Tensor,
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
        # Validate input shape
        if len(inp.shape) != 4:
            raise ValueError("Input tensor must have shape (batch_size, channels, height, width)")

        # bring everything together
        batch_size, channels, height, width = inp.shape
        inp_flat = inp.permute(1,0,2,3).contiguous().view(channels, -1, 1)

        # flatten mask
        mask_flat = mask.view(-1)
        
        # initialize the output
        out_flat = torch.zeros((out_channels, batch_size*height*width,1), device=inp.device, dtype=inp.dtype)

        # form together the output
        out_flat[ :, mask_flat] = module(inp_flat[:, mask_flat])
        
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


    def define_urban_extractor(self) -> None:
        """
        Define the urban extractor if not already defined
        """
        
        if not hasattr(self, "building_extractor"):
            print("Loading urban extractor")
            MODEL = Namespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[8, 16,] )
            CONSISTENCY_TRAINER = Namespace(LOSS_FACTOR=0.5)
            PATHS = Namespace(OUTPUT="model/DDA_model/checkpoints/")
            DATALOADER = Namespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
            TRAINER = Namespace(LR=1e5)
            cfg = Namespace(MODEL=MODEL, CONSISTENCY_TRAINER=CONSISTENCY_TRAINER, PATHS=PATHS, DATALOADER=DATALOADER, TRAINER=TRAINER, NAME=f"fusionda_newAug8_16")
            self.building_extractor, _, _ = load_checkpoint(epoch=30, cfg=cfg, device="cuda", no_disc=True)
            self.building_extractor = self.building_extractor.cuda()



    def create_building_score(self, inputs: dict) -> torch.Tensor:
        """
        input:
            - inputs: dictionary with the input data
        output:
            - score: building score
        """

        # initialize the neural network, load from checkpoint
        self.define_urban_extractor()
        self.building_extractor.eval()
        self.unetmodel.freeze_bn_layers()
 
        # add padding
        X, (px1,px2,py1,py2) = self.add_padding(inputs["input"], True)

        # forward the neural network
        # with torch.no_grad():
        if self.S1 and self.S2:
            X = torch.cat([
                X[:, 4:6], # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
            dim=1)
            _, _, logits, _, _ = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif self.S1 and not self.S2:
            X = torch.cat([
                X, # S1
                torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
            dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif not self.S1 and self.S2:
            X = torch.cat([
                torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
            dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
            
        # forward the model
        # _, _, logits, _, _ = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        score = torch.sigmoid(logits)

        # revert padding
        score = self.revert_padding(score, (px1,px2,py1,py2))

        return score
    
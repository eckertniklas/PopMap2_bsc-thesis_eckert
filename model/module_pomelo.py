 
import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
# from model.customUNet import CustomUNet
# import ast

from model.DDA_model.utils.networks import load_checkpoint

from utils.plot import plot_2dmatrix, plot_and_save
# from utils.utils import Namespace
from utils.constants import dda_cfg, stage1feats, stage2feats

class POMELO_module(nn.Module):
    '''
    PomeloUNet
    Description:
        - UNet with a regression head

    '''
    def __init__(self, input_channels, feature_extractor="DDA", down=5,
                occupancymodel=False, pretrained=False, head="v3",
                sparse_unet=False, buildinginput=True, biasinit=0.75,
                sentinelbuildings=True):
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
        """

        self.down = down
        self.occupancymodel = occupancymodel
        self.buildinginput = buildinginput
        self.feature_extractor = feature_extractor
        self.sparse_unet = sparse_unet
        self.sentinelbuildings = sentinelbuildings

        self.head_name = head 
        head_input_dim = 0
        
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

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

            ## load weights from checkpoint
            self.unetmodel, _, _ = load_checkpoint(epoch=30, cfg=dda_cfg, device="cuda", no_disc=True)

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
            raise ValueError("feature_extractor {} not supported".format(feature_extractor))

        # if buildinginput:
        #     head_input_dim += 1

        if head=="v3":
            h = 64
            head_input_dim += unet_out
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(h, 2, kernel_size=1, padding=0)
            )
        
        elif head=="v3_slim":
            h = 64
            head_input_dim += unet_out
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, 2, kernel_size=1, padding=0),
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

        #HACK: change bias for unetmodel.fusion_out_conv
        # buildinghead_bias = torch.tensor([-3.0])
        # self.unetmodel.fusion_out_conv.conv.bias = torch.nn.parameter.Parameter(data=buildinghead_bias, requires_grad=True)

        # define urban extractor, which is again a dual stream unet
        print("Loading urban extractor")
        self.building_extractor, _, _ = load_checkpoint(epoch=30, cfg=dda_cfg, device="cuda", no_disc=True)
        self.building_extractor = self.building_extractor.cuda()


# NEW FORWARD
    def forward(self, inputs, train=False, padding=True, return_features=True,
                encoder_no_grad=False, unet_no_grad=False, sparse=False,
                builtuploss=False, basicmethod=False, twoheadmethod=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """

        """
            - builtuploss = True: activates builtup-loss add-on
        """

        X = inputs["input"]

        # create building score, if not available in the dataset, or overwrite it if sentinelbuildings or builtuploss is True
        if "building_counts" not in inputs.keys() or self.sentinelbuildings or builtuploss:
            if builtuploss and basicmethod:
                inputs["building_counts"]  = self.create_building_score(inputs, builtuploss=builtuploss, basicmethod=basicmethod)
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    inputs["building_counts"]  = self.create_building_score(inputs, builtuploss=builtuploss, basicmethod=basicmethod)
                torch.cuda.empty_cache()

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

        # if self.buildinginput:
        #     # append building counts to the middle features
        #     middlefeatures.append(inputs["building_counts"])

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

        headin = torch.cat(middlefeatures, dim=1)

        # forward the head
        if sparse:
            out = self.sparse_module_forward(headin, sparsity_mask, self.head, out_channels=2)[:,0]
        else:
            out = self.head(headin)[:,0]

        # forward the builtup head for the twoheadmethod
        if twoheadmethod:
            if sparse:
                out_bu = self.sparse_module_forward(headin, sparsity_mask, self.unetmodel.fusion_out_conv, out_channels=1)#[:,0]
            else:
                out_bu = self.unetmodel.fusion_out_conv(headin)#[:,0]

        # Population map and total count
        if self.occupancymodel:

            # activation function for the population map is a ReLU to avoid negative values
            scale = nn.functional.relu(out)
            # activation function for builtup score is sigmoid to get probability values
            if twoheadmethod:
                # subtract_val = 0.5
                # out_bu = torch.subtract(out_bu, subtract_val)
                score_bu = nn.functional.sigmoid(out_bu)

            if "building_counts" in inputs.keys():
                
                # save the scale
                if sparse and self.sparse_unet:
                    aux["scale"] = torch.cat( [(scale* ratio.view(ratio.shape[0],1,1))[subsample_mask_empty] ,
                                                   scale[building_sparsity_mask] ]  , dim=0)
                elif sparse:
                    aux["scale"] = scale[sparsity_mask]
                else:
                    aux["scale"] = scale

                # Get the population density map
                if builtuploss and twoheadmethod:
                    popdensemap = scale * score_bu[:,0]
                else:
                    popdensemap = scale * inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys(), but occupancy model is True")
        else:
            popdensemap = nn.functional.relu(out)
            aux["scale"] = None
        
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


        
        if builtuploss and basicmethod:
            return {"popcount": popcount, "popdensemap": popdensemap, "builtup_score": inputs["building_counts"],
                    **aux,
                    }
        if builtuploss and twoheadmethod:
            return {"popcount": popcount, "popdensemap": popdensemap, "builtup_score": inputs["building_counts"], "twohead_builtup_score": score_bu,
                    **aux,
                    }
        else:  
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


    def create_building_score(self, inputs: dict, builtuploss=False, basicmethod=False) -> torch.Tensor:
        """
        input:
            - inputs: dictionary with the input data
        output:
            - score: building score
        """

        # initialize the neural network, load from checkpoint
        if basicmethod is False:
            self.building_extractor.eval()
        self.unetmodel.freeze_bn_layers()

        """
        if builtuploss:
            self.unetmodel.freeze_bn_layers()
        else:
            self.building_extractor.eval()
            self.unetmodel.freeze_bn_layers()
        """

        # add padding
        X, (px1,px2,py1,py2) = self.add_padding(inputs["input"], True)

        # forward the neural network
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
        score = torch.sigmoid(logits)

        # revert padding
        score = self.revert_padding(score, (px1,px2,py1,py2))

        return score
    
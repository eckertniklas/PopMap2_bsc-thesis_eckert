


import torch.nn as nn
import torch
import torch.nn.functional as F

# import copy
import segmentation_models_pytorch as smp

from utils.plot import plot_2dmatrix

import torch.nn as nn


class CustomUNet(smp.Unet):
    def __init__(self, encoder_name, in_channels, classes, down=3, fsub=16, pretrained=False,
                 dilation=1, replace7x7=True, activation=nn.LeakyReLU, grouped=False):

        # instanciate the base model
        # super().__init__(encoder_name, encoder_weights="imagenet",
        # super().__init__(encoder_name, encoder_weights="swsl",
        super().__init__(encoder_name, encoder_weights="imagenet" if pretrained else None,
                        in_channels=in_channels, classes=classes, decoder_channels=(64,32,16), 
                        decoder_use_batchnorm=False, encoder_depth=3, activation=activation)
        
        self.decoder_channels = (256,128,64,32,16)[-down:] 
        self.latent_dim = sum(self.decoder_channels)
        self.fsub = fsub

        # Adjust the U-Net depth to 2 or lower here
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=down,
            weights="imagenet" if pretrained else None,
        )

        self.decoder = smp.decoders.unet.model.UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=down,
            use_batchnorm=False,
            center=True if encoder_name.startswith("vgg") else False
        )

        if dilation > 1:
            self.modify_dilation(self.encoder, dilation=dilation)

        # replace first layer with 3x3 conv instead of 7x7 (better suitable for remote sensing datasets)
        if encoder_name.startswith("resnet") and replace7x7:
            conv1w = self.encoder.conv1.weight # old kernel
            self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False, dilation=1)
            self.encoder.conv1.weight = nn.Parameter(conv1w[:,:,2:-2,2:-2])
            kernel_size, padding = 3, 1
        else:
            kernel_size, padding = 7, 3
            if encoder_name.startswith("resnet"):
                conv1w = self.encoder.conv1.weight # old kernel

                # we have to revert the dilation that might have been applied before
                if dilation > 1:
                    # self.encoder.conv1.modify_dilation(dilation=1)
                    self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, dilation=1)
                    self.encoder.conv1.weight = nn.Parameter(conv1w)
            else:
                conv1w = self.encoder.features[0]
                if dilation > 1:
                    self.encoder.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, dilation=1)
                    self.encoder.features[0].weight = nn.Parameter(conv1w)

        if grouped:
            # replace first layer with grouped convolutions to avoid early interaction
            conv1w = self.encoder.conv1.weight # old kernel
            self.encoder.conv1 = CustomGroupedConvolution(sentinel_2_channels=4, sentinel_1_channels=2,
                                                          out_channels=64,
                                                          kernel_size=kernel_size, padding=padding, bias=False, stride=2)
            self.encoder.conv1.s2_conv.weight = nn.Parameter(conv1w[:64//2, :4, :, :])
            self.encoder.conv1.s1_conv.weight = nn.Parameter(conv1w[64//2:, 4:, :, :])

            # replace other convolutions at the second level as well
            self.encoder.layer1[0].conv1 = self.replace_with_groups(self.encoder.layer1[0].conv1) # old kernel
            self.encoder.layer1[0].conv2 = self.replace_with_groups(self.encoder.layer1[0].conv2) # old kernel
            self.encoder.layer1[1].conv1 = self.replace_with_groups(self.encoder.layer1[1].conv1) # old kernel
            self.encoder.layer1[1].conv2 = self.replace_with_groups(self.encoder.layer1[1].conv2) # old kernel

            self.encoder.layer2[0].conv1 = self.replace_with_groups(self.encoder.layer2[0].conv1, stride=2) # old kernel
            self.encoder.layer2[0].conv2 = self.replace_with_groups(self.encoder.layer2[0].conv2) # old kernel
            self.encoder.layer2[1].conv1 = self.replace_with_groups(self.encoder.layer2[1].conv1) # old kernel
            self.encoder.layer2[1].conv2 = self.replace_with_groups(self.encoder.layer2[1].conv2) # old kernel
            
        # adapt size of the center block for vgg
        if encoder_name.startswith("vgg"):
            self.decoder.center = smp.decoders.unet.decoder.CenterBlock(
                in_channels=self.encoder.out_channels[-1],
                out_channels=self.encoder.out_channels[-1]
            )
        else:
            self.decoder.center = nn.Identity()
 
        self.remove_batchnorm(self)

        # initialize
        print("self.encoder.out_channels", self.encoder.out_channels)
        self.name = "u-{}".format(encoder_name)
        self.initialize()

        # print number of parameters that are actually used
        self.num_params = self.num_effective_params(down=down, verbose=True)


    def remove_batchnorm(self, model):
        """
        remove batchnorm layers from model
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, nn.Identity())
            else:
                self.remove_batchnorm(module)


    def replace_with_groups(self, module, kernel_size=3, padding=1, stride=1):
        convXw = module.weight # old kernel
        module = CustomGroupedConvolution(sentinel_2_channels=convXw.shape[1]//2, sentinel_1_channels=convXw.shape[1]//2,
                                                            out_channels=convXw.shape[0],
                                                            kernel_size=kernel_size, padding=padding, bias=False, stride=stride)
        module.s2_conv.weight = nn.Parameter(convXw[:convXw.shape[0]//2, :convXw.shape[1]//2, :, :])
        module.s1_conv.weight = nn.Parameter(convXw[convXw.shape[0]//2:, convXw.shape[1]//2:, :, :])
        return module


    def modify_dilation(self, net, dilation=2):
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                module.dilation = (dilation, dilation)
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                padding_needed = (kernel_size - 1)
                module.padding = (padding_needed, padding_needed)


    def num_effective_params(self, down=5, verbose=False):
        """
        print number of parameters that are actually used
        """
        stages_param_count = 0

        # encoder params
        for i,el in enumerate(self.encoder.get_stages()[:down+1]):
            this_stage_sum = sum(p.numel() for p in el.parameters() if p.requires_grad)
            stages_param_count += this_stage_sum
            if verbose:
                print("stage", i, ":", this_stage_sum)
        
        # decoder and segmentation head
        decoder_sum = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        segmentation_sum = sum(p.numel() for p in self.segmentation_head.parameters() if p.requires_grad)
        if verbose:
            print("decoder:", decoder_sum)
            print("segmentation_head:", segmentation_sum)
            print("Total # of effective Parameters:", stages_param_count+decoder_sum+segmentation_sum)
        return stages_param_count+decoder_sum+segmentation_sum


    def forward(self, x, return_features=True, encoder_no_grad=False):
        """
        pass `x` trough model`s encoder, decoder and heads
        """

        self.check_input_shape(x)
        bs,_,h,w = x.shape

        # encoder
        if encoder_no_grad:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        # rearrange features to start from head of encoder
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.decoder.center(head)

        # decoder, with skip connections
        decoder_features = []
        for i, decoder_block in enumerate(self.decoder.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

            if return_features:
                # rs, re = torch.randint(self.fsub,self.fsub*2,(1,)).item(), torch.randint(self.fsub*2,self.fsub*2,(1,)).item()
                xup = F.interpolate(x, size=(h//self.fsub,w//self.fsub), mode='nearest') # interpolate/subsample to other size
                decoder_features.append(xup.view(bs,x.size(1),-1))
        decoder_output = x
        
        # bunddle features
        if return_features:
            decoder_features = torch.concatenate(decoder_features,1)

        # segmentation head forward
        masks = self.segmentation_head(decoder_output)

        # return mask and labels if classification head is present
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, decoder_features
    

class CustomGroupedConvolution(nn.Module):
    def __init__(self, sentinel_2_channels, sentinel_1_channels, out_channels,
                 kernel_size=3, padding=1, bias=False, stride=2):
        super(CustomGroupedConvolution, self).__init__()

        # assert that the number of channels is divisible by 2
        assert out_channels % 2 == 0, "The number of output channels should be divisible by 2"
        self.s2c = sentinel_2_channels
        
        # Sentinel-2 Convolution
        self.s2_conv = nn.Conv2d(sentinel_2_channels, out_channels//2, kernel_size=kernel_size,
                                 padding=padding, bias=bias, stride=stride)
        
        # Sentinel-1 Convolution
        self.s1_conv = nn.Conv2d(sentinel_1_channels, out_channels//2, kernel_size=kernel_size,
                                 padding=padding, bias=bias, stride=stride)

    def forward(self, x):
        # Splitting the input based on the channels
        s2_data = x[:, :self.s2c, :, :]  # Taking the first :x channels for Sentinel-2
        s1_data = x[:, self.s2c:, :, :]  # Taking the next x: channels for Sentinel-1
        
        # Grouped Convolutions
        s2_out = self.s2_conv(s2_data)
        s1_out = self.s1_conv(s1_data)
        
        # Concatenate along the channel dimension
        out = torch.cat((s2_out, s1_out), dim=1)
        
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from collections import OrderedDict

from pathlib import Path

from ..utils import experiment_manager

from utils.plot import plot_2dmatrix
from tqdm import tqdm

def create_network(cfg):
    return DualStreamUNet(cfg) if cfg.MODEL.TYPE == 'dualstreamunet' else UNet(cfg)


def save_checkpoint(network, optimizer, epoch: int, step: int, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}_lossweight{cfg.CONSISTENCY_TRAINER.LOSS_FACTOR}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch: int, cfg: experiment_manager.CfgNode, device, no_disc):
    net = create_network(cfg)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}_lossweight{cfg.CONSISTENCY_TRAINER.LOSS_FACTOR}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    checkpoint['network']

    if no_disc:
        net.load_state_dict(checkpoint['network'], strict=False)
    else:
        net.load_state_dict(checkpoint['network'])

    # net.load_state_dict(checkpoint['network'])
    # optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']

class ownDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ownDiscriminator, self).__init__()
        
        #Receptive Field: 270 for 4 conv layer, for only two conv layers it is 18x18, 1 conv layer has 7x7
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        #x3 = self.relu(self.conv3(x2))
        #x4 = self.relu(self.conv4(x3))
        
        # Decoder
        #x = self.relu(self.deconv1(x2))
        #x = self.relu(self.deconv2(x + x1))
        x = self.relu(self.deconv3(x2))
        x = self.sigmoid(self.deconv4(x + x1))
        
        return x


class UNet(nn.Module):
    def __init__(self, cfg, n_channels=None, n_classes=None, topology=None, enable_outc=True):

        self._cfg = cfg

        n_channels = cfg.MODEL.IN_CHANNELS if n_channels is None else n_channels
        n_classes = cfg.MODEL.OUT_CHANNELS if n_classes is None else n_classes
        topology = cfg.MODEL.TOPOLOGY if topology is None else topology

        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]  # topography upwards
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]  # last layer

            layer = Down(in_dim, out_dim, DoubleConv)

            print(f'down{idx + 1}: in {in_dim}, out {out_dim}')
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]

            layer = Up(in_dim, out_dim, DoubleConv)

            print(f'up{idx + 1}: in {in_dim}, out {out_dim}')
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x1, x2=None, encoder_no_grad=False):
        x = x1 if x2 is None else torch.cat((x1, x2), 1)

        if encoder_no_grad:
            with torch.no_grad():
                x1 = self.inc(x)

                inputs = [x1]
                # Downward U:
                for layer in self.down_seq.values():
                    out = layer(inputs[-1])
                    inputs.append(out)
        else:  
            x1 = self.inc(x)

            inputs = [x1]
            # Downward U:
            for layer in self.down_seq.values():
                out = layer(inputs[-1])
                inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)  # x1 for next up layer

        out = self.outc(x1) if self.enable_outc else x1

        return out


class DualStreamUNet(nn.Module):

    def __init__(self, cfg):
        super(DualStreamUNet, self).__init__()
        self._cfg = cfg
        out = cfg.MODEL.OUT_CHANNELS
        topology = cfg.MODEL.TOPOLOGY
        out_dim = topology[0]

        # sentinel-1 sar unet stream
        sar_in = len(cfg.DATALOADER.SENTINEL1_BANDS)
        self.sar_stream = UNet(cfg, n_channels=sar_in, n_classes=out, topology=topology, enable_outc=False)
        self.sar_in = sar_in
        self.sar_out_conv = OutConv(out_dim, out)

        # sentinel-2 optical unet stream
        optical_in = len(cfg.DATALOADER.SENTINEL2_BANDS)
        self.optical_stream = UNet(cfg, n_channels=optical_in, n_classes=out, topology=topology, enable_outc=False)
        self.optical_in = optical_in
        self.optical_out_conv = OutConv(out_dim, out)

        # out block combining unet outputs
        fusion_out_dim = 2 * out_dim 
        self.fusion_out_conv = OutConv(fusion_out_dim, out)

        #Discriminator
        self.disc = ownDiscriminator(in_channels=out_dim, out_channels=2)

        self.patchsize = 512 # self.sar_streamself.sar_streamfor patchwise inference

    def freeze_bn_layers(self):
        for _, layer in self.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False


    def forward(self, x_fusion, alpha=0, encoder_no_grad=False, return_features=False, S1=True, S2=True):

        features = []

        # sar
        if S1:
            features_sar = self.sar_stream(x_fusion[:, :self.sar_in, ], encoder_no_grad=encoder_no_grad)
            features.append(features_sar)

        # optical
        if S2:
            features_optical = self.optical_stream(x_fusion[:, self.sar_in:, ], encoder_no_grad=encoder_no_grad)
            features.append(features_optical)

        # features_fusion = torch.cat((features_sar, features_optical), dim=1)
        features_fusion = torch.cat(features, dim=1)
        if return_features:
            return features_fusion
        logits_fusion = self.fusion_out_conv(features_fusion)

        #### get features before outConv 
        #stacked_features = torch.cat((features_sar, features_optical), dim=0)
        if alpha != 0:
            reverse_feature_sar = ReverseLayerF.apply(features_sar, alpha)
            reverse_feature_optical = ReverseLayerF.apply(features_optical, alpha)
            logits_disc_sar = self.disc(reverse_feature_sar)
            logits_disc_optical = self.disc(reverse_feature_optical)
        else:
            logits_disc_sar = None
            logits_disc_optical = None

        logits_sar = self.sar_out_conv(features_sar)
        logits_optical = self.optical_out_conv(features_optical)

        if return_features:
            return logits_sar, logits_optical, logits_fusion, logits_disc_sar, logits_disc_optical, features_fusion
        else:
            return logits_sar, logits_optical, logits_fusion, logits_disc_sar, logits_disc_optical

    def fusion_features(self, x_fusion):

        # sar
        # x_sar = x_fusion[:, :self.sar_in, ]
        features_sar = self.sar_stream(x_fusion[:, :self.sar_in, ])

        # optical
        # x_optical = x_fusion[:, self.sar_in:, ]
        features_optical = self.optical_stream(x_fusion[:, self.sar_in:, ])

        features_fusion = torch.cat((features_sar, features_optical), dim=1)
        return features_fusion


    def sparse_forward(self, x: torch.tensor, sparsity_mask, alpha=0, return_features=True, encoder_no_grad=False, S1=True, S2=True) -> torch.Tensor:
        """
        patchwise forward pass
        """

        # self.check_input_shape(x)
        # assert not return_features, "return_features is not supported for sparse_forward"

        # pad the input tensor to be divisible by the patchsize
        bs,_,h,w = x.shape
        pad_h = self.patchsize - h%self.patchsize
        pad_w = self.patchsize - w%self.patchsize
        x = F.pad(x, (0,pad_w,0,pad_h), mode='constant', value=0)
        sparsity_mask = F.pad(sparsity_mask, (0,pad_w,0,pad_h), mode='constant', value=0)

        # initialize output
        bs,_,hnew,wnew = x.shape
        out = torch.zeros((bs,self.sar_stream.up_seq.up2.conv.conv[3].out_channels*2,hnew,wnew), device=x.device)
        
        overlap = 16  # Size of the overlapping region on each edge
        stride = self.patchsize - 2*overlap  # Assuming a stride of 32; you can adjust this as per your requirement 

        # divide the input tensor x into a grid and iterate over patches
        for i in range(0, hnew, stride):
            for j in range(0, wnew, stride):
                # Compute actual patch boundaries, considering overlap and image edges
                i1 = max(i - overlap, 0)
                i2 = min(i + self.patchsize + overlap, hnew)
                j1 = max(j - overlap, 0)
                j2 = min(j + self.patchsize + overlap, wnew)

                mask_patch = sparsity_mask[:, i1:i2, j1:j2]
                active_idxs = torch.nonzero(torch.sum(mask_patch, dim=(1, 2)) > 0)

                if active_idxs.nelement() > 0:
                    x_patch = x[active_idxs[:, 0], :, i1:i2, j1:j2]

                    # Forward pass
                    out_patch = self.forward(x_patch, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=S1, S2=S2)
                    # out_patch = self.forward(x_patch, return_features=True, encoder_no_grad=encoder_no_grad)[0]

                    # Add the patch to the output tensor
                    i1_n = i1 + overlap if i1 != 0 else i1
                    i2_n = i2 - overlap if i2 != h else i2
                    j1_n = j1 + overlap if j1 != 0 else j1
                    j2_n = j2 - overlap if j2 != w else j2

                    out[active_idxs[:, 0], :, i1_n:i2_n, j1_n:j2_n] = out_patch[:, :, i1_n - i1:i2_n - i1, j1_n - j1:j2_n - j1]

        return out[:, :, :h, :w]
    

# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

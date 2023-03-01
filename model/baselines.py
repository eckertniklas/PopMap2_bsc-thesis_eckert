
'''
Adapted from Pytorch ResNet Implementation
'''
from torch import nn

from utils.constants import osm_features
import torchvision.models as models


class EOResNet(nn.Module):
    '''
    Earth Observation ResNet 50
    '''
    def __init__(self, input_channels, num_classes):
        super(EOResNet, self).__init__()

        ## set model features
        self.model = models.resnet50(pretrained=False) # pretrained=False just for debug reasons
        first_conv_layer = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        first_conv_layer = [first_conv_layer]
        

        first_conv_layer.extend(list(self.model.children())[1:-1])
        self.model = nn.Sequential(*first_conv_layer) 

        self.clf_layer = nn.Linear(in_features=2048, out_features=num_classes)
        self.clf_layer.apply(init_weights)        

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        outputs = self.clf_layer(outputs)
        return outputs
    


class EO2ResNet_OSM(nn.Module):
    '''
    Also using OSM Data
    '''
    def __init__(self, input_channels, num_classes, scale_factor=1):
        super(EO2ResNet_OSM, self).__init__()
        self.scale_factor = scale_factor
        num_features = [1024, 2048, 4096]
        self.cnn = EOResNet(input_channels, num_classes).model
        self.linear = linear_resnet50(scale_factor=scale_factor)
        if self.scale_factor != 1:
            self.lin_scale = nn.Sequential(
                nn.BatchNorm1d(num_features[1]*scale_factor),
                nn.Linear(num_features[1]*scale_factor, num_features[1]),
            )

        ## final depends on cnn output concat with linear output
        self.final = nn.Sequential(
            nn.Linear(in_features=num_features[2], out_features=num_features[1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=num_features[1], out_features=num_features[0]),
            nn.LeakyReLU(inplace=True)
            )
        self.clf_layer = nn.Linear(in_features=num_features[0],out_features=num_classes)
        self.final.apply(init_weights)
        self.clf_layer.apply(init_weights)

    def forward(self, inputs, osm_in):
        outputs = self.cnn(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        osm_in = osm_in.view(osm_in.shape[0],-1)
        osm_out = self.linear(osm_in)
        if self.scale_factor != 1:
            osm_out = self.lin_scale(osm_out)
        outputs = torch.cat((outputs, osm_out), dim=1)
        outputs = self.final(outputs)
        outputs = self.clf_layer(outputs)
        return outputs


def init_weights(layer, method = 'xavier normal'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == 'xavier normal':
            nn.init.xavier_normal_(layer.weight)
        elif method == 'kaiming normal':
            nn.init.kaiming_normal_(layer.weight)




class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.lin1 = nn.Linear(inplanes, width)
        self.bn1 = norm_layer(width)
        self.lin2 = nn.Linear(width, width)
        self.bn2 = norm_layer(width)
        self.lin3 = nn.Linear(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.lin2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.lin3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class LinearResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, scale_factor=1):
        super(LinearResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64*scale_factor
        self.dimension = [self.inplanes, self.inplanes*2, self.inplanes*4, self.inplanes*8]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.lin1 = nn.Linear(3, self.inplanes)
        self.lin1 = nn.Linear(osm_features, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.dimension[0], layers[0])
        self.layer2 = self._make_layer(block, self.dimension[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.dimension[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.dimension[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Linear(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = LinearResNet(block, layers, **kwargs)
    if pretrained:
        pass
    return model

def linear_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



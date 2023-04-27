'Code from Official ProtoPNet implemetation: https://github.com/cfchen-duke/ProtoPNet'


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy
from .lrp_general6 import sum_stacked2, get_lrpwrapperformodule, bnafterconv_overwrite_intoconv, resetbn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_dir = './pretrained_models'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # only conv with possibly not 1 stride
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, 3, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, 1, 0]

        return block_kernel_sizes, block_strides, block_paddings


class ResNet_features(nn.Module):
    '''
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    '''

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_features, self).__init__()

        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network, not counting the number
        of bypass layers
        '''

        return (self.block.num_layers * self.layers[0]
              + self.block.num_layers * self.layers[1]
              + self.block.num_layers * self.layers[2]
              + self.block.num_layers * self.layers[3]
              + 1)


    def __repr__(self):
        template = 'resnet{}_features'
        return template.format(self.num_layers() + 1)

def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet101'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet152'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

class Modulenotfounderror(Exception):
    pass


class BasicBlock_fused(BasicBlock):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_fused, self).__init__(inplanes, planes, stride, downsample)
        self.elt = sum_stacked2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))
        out = self.relu(out)
        return out


class Bottleneck_fused(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_fused, self).__init__(inplanes, planes, stride, downsample)
        self.elt = sum_stacked2()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.elt(torch.stack([out, identity], dim=0))
        out = self.relu(out)
        return out

class ResNet_canonized(ResNet_features):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_canonized, self).__init__(block, layers, num_classes=1000, zero_init_residual=False)
    def setbyname(self, name, value):
        def iteratset(obj, components, value):

            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)
                return True
            else:
                nextobj = getattr(obj, components[0])
                return iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = iteratset(self, components, value)
        return success

    def copyfrom(self, net, lrp_params, lrp_layer2method, verbose: bool = False):
        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            if isinstance(src_module, nn.Linear):
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)

            elif isinstance(src_module, nn.Conv2d):
                # store conv2d layers
                last_src_module_name = src_module_name
                last_src_module = src_module

            if isinstance(src_module, nn.BatchNorm2d):
                thisis_inputconv_andiwant_zbeta = lrp_params['use_zbeta'] and (last_src_module_name == 'conv1')
                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                # wrap conv
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)

                if not self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")
                updated_layers_names.append(last_src_module_name)

                # wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)

        # sum_stacked2 is present only in the targetclass, so must iterate here
        for target_module_name, target_module in self.named_modules():
            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + target_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

            elif isinstance(target_module, sum_stacked2):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + target_module_name + " in target net , impossible!")
                updated_layers_names.append(target_module_name)

        if verbose:
            for target_module_name, target_module in self.named_modules():
                if target_module_name not in updated_layers_names:
                    print('not updated:', target_module_name)


def _resnet_canonized(arch, block, layers, **kwargs):
    model = ResNet_canonized(block, layers, **kwargs)
    return model


def resnet18_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet18', BasicBlock_fused, [2, 2, 2, 2], **kwargs)


def resnet50_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet50', Bottleneck_fused, [3, 4, 6, 3], **kwargs)


def resnet34_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet34', BasicBlock_fused, [3, 4, 6, 3], **kwargs)


def resnet152_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet152', Bottleneck_fused, [3, 8, 36, 3], **kwargs)


def resnet101_canonized(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_canonized('resnet101', Bottleneck_fused, [3, 4, 23, 3], **kwargs)

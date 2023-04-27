'Code from Official ProtoPNet implemetation: https://github.com/cfchen-duke/ProtoPNet'


import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .lrp_general6 import get_lrpwrapperformodule, bnafterconv_overwrite_intoconv, resetbn
import copy

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_dir = './pretrained_models'

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_features(nn.Module):

    def __init__(self, cfg, batch_norm=False, init_weights=True):
        super(VGG_features, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features = self._make_layers(cfg, batch_norm)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm):

        self.n_layers = 0

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)

                in_channels = v

        return nn.Sequential(*layers)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    def __repr__(self):
        template = 'VGG{}, batch_norm={}'
        return template.format(self.num_layers() + 3,
                               self.batch_norm)



def vgg11_features(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg11'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg11_bn_features(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg11_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg13_features(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg13'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg13_bn_features(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg13_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg16_features(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg16'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg16_bn_features(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg16_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg19_features(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg19'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg19_bn_features(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg19_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


class VGGCanonized(VGG_features):

    def __init__(self, config, batch_norm, pretrained=False):
        super(VGGCanonized, self).__init__(config, batch_norm)

    def setbyname(self, name, value) -> bool:
        """ Find and replace attribute inside this object

        :param name: Attribute name
        :param value: Attibute value
        :returns: True if attribute was found and replaced, False otherwise
        """
        def iteratset(obj, components, value) -> bool:

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

    def copyfrom(self, net, lrp_params, lrp_layer2method, verbose: bool = True):
        """ Copy layer parameters and wrap everything for LRP

        :param net: Source network
        :param lrp_params: LRP rules
        :param lrp_layer2method: Replacement layers
        """
        class Modulenotfounderror(Exception):
            pass

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        for src_module_name, src_module in net.named_modules():
            if isinstance(src_module, nn.Linear):
                # copy linear layers
                wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)

            if isinstance(src_module, nn.Conv2d):
                last_src_module_name = src_module_name
                last_src_module = src_module

            if isinstance(src_module, nn.BatchNorm2d):
                # Detect input convolution
                thisis_inputconv_andiwant_zbeta = lrp_params['use_zbeta'] and (last_src_module_name == 'features.0')
                # Wrap convolution
                m = copy.deepcopy(last_src_module)
                m = bnafterconv_overwrite_intoconv(m, bn=src_module)
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)
                if not self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")
                updated_layers_names.append(last_src_module_name)

                # Wrap batchnorm
                wrapped = get_lrpwrapperformodule(resetbn(src_module), lrp_params, lrp_layer2method)
                if not self.setbyname(src_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + src_module_name + " in target net to copy")
                updated_layers_names.append(src_module_name)

            if isinstance(src_module, nn.ReLU):
                # Detect input convolution
                thisis_inputconv_andiwant_zbeta = lrp_params['use_zbeta'] and (last_src_module_name == 'features.0')
                # Wrap convolution
                m = copy.deepcopy(last_src_module)
                wrapped = get_lrpwrapperformodule(m, lrp_params, lrp_layer2method,
                                                  thisis_inputconv_andiwant_zbeta=thisis_inputconv_andiwant_zbeta)
                if not self.setbyname(last_src_module_name, wrapped):
                    raise Modulenotfounderror(
                        "could not find module " + last_src_module_name + " in target net to copy")
                updated_layers_names.append(last_src_module_name)

        # Wrapped activation and pooling layers
        for target_module_name, target_module in self.named_modules():
            if isinstance(target_module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                wrapped = get_lrpwrapperformodule(target_module, lrp_params, lrp_layer2method)
                if not self.setbyname(target_module_name, wrapped):
                    raise Modulenotfounderror("could not find module " + target_module_name + " in target net to copy")
                updated_layers_names.append(target_module_name)

        if verbose:
            for target_module_name, target_module in self.named_modules():
                if target_module_name not in updated_layers_names:
                    if not target_module_name.endswith('.module'):
                        print('not updated:', target_module_name)


def vgg11_canonized(**kwargs):
    return VGGCanonized(cfg['A'], batch_norm=False)


def vgg11_bn_canonized(**kwargs):
    return VGGCanonized(cfg['A'], batch_norm=True)


def vgg13_canonized(**kwargs):
    return VGGCanonized(cfg['B'], batch_norm=False)


def vgg13_bn_canonized(**kwargs):
    return VGGCanonized(cfg['B'], batch_norm=True)


def vgg16_canonized(**kwargs):
    return VGGCanonized(cfg['D'], batch_norm=False)


def vgg16_bn_canonized(**kwargs):
    return VGGCanonized(cfg['D'], batch_norm=True)


def vgg19_canonized(**kwargs):
    return VGGCanonized(cfg['E'], batch_norm=False)


def vgg19_bn_canonized(**kwargs):
    return VGGCanonized(cfg['E'], batch_norm=True)

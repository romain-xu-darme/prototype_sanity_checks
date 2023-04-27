from prototree.prototree import ProtoTree
from features.lrp_general6 import *
from features.resnet_features import *
from features.vgg_features import *

base_architecture_to_features = {'resnet18': resnet18_canonized,
                                 'resnet34': resnet34_canonized,
                                 'resnet50': resnet50_canonized,
                                 'resnet101': resnet101_canonized,
                                 'resnet152':resnet152_canonized,
                                 'vgg11': vgg11_canonized,
                                 'vgg11_bn': vgg11_bn_canonized,
                                 'vgg13': vgg13_canonized,
                                 'vgg13_bn': vgg13_bn_canonized,
                                 'vgg16': vgg16_canonized,
                                 'vgg16_bn': vgg16_bn_canonized,
                                 'vgg19': vgg19_canonized,
                                 'vgg19_bn': vgg19_bn_canonized,
                                 }


def canonize_tree(tree: ProtoTree, arch: str, device: str) -> ProtoTree:
    """ Prepare ProtoTree for PRP pass

    :param tree: ProtoTree
    :param arch: Backbone architecture
    :returns: modified tree
    """
    lrp_params_def1 = {
        'conv2d_ignorebias': True,
        'eltwise_eps': 1e-6,
        'linear_eps': 1e-6,
        'pooling_eps': 1e-6,
        'use_zbeta': True,
    }

    lrp_layer2method = {
        'nn.ReLU': relu_wrapper_fct,
        'nn.Sigmoid': relu_wrapper_fct,
        'nn.BatchNorm2d': relu_wrapper_fct,
        'nn.Conv2d': conv2d_beta0_wrapper_fct,
        'nn.Linear': linearlayer_eps_wrapper_fct,
        'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
        'nn.MaxPool2d': maxpool2d_wrapper_fct,
        'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct
    }
    assert arch in base_architecture_to_features.keys(), f'Unsupported architecture {arch}'
    backbone = base_architecture_to_features[arch](pretrained=False)
    backbone.copyfrom(tree._net,
                      lrp_params=lrp_params_def1,
                      lrp_layer2method=lrp_layer2method)
    tree._net = backbone.to(device)
    wrapped_add_on = [get_lrpwrapperformodule(
        copy.deepcopy(src_module),
        lrp_params_def1, lrp_layer2method)
                         for _, src_module in tree._add_on.named_modules() if not isinstance(src_module, nn.Sequential)]
    tree._add_on = nn.Sequential(*wrapped_add_on)
    # Add fields for compatibility with ProtoPNet
    tree.max_layer = get_lrpwrapperformodule(torch.nn.MaxPool2d((7, 7), return_indices=False),
                                             lrp_params_def1, lrp_layer2method)
    tree.ones = nn.Parameter(torch.ones_like(tree.prototype_layer.prototype_vectors),
                             requires_grad=False)
    tree.epsilon = 1e-4
    return tree.to(device)

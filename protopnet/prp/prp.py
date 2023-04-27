#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 12:43:52 2020

@author: srishtigautam

Code built upon LRP code from https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet
"""

from __future__ import print_function, division

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .resnet_features import *
from .vgg_features import *
from .heatmaphelpers import *
from .lrp_general6 import *

class addon_canonized(nn.Module):

    def __init__(self):
        super(addon_canonized, self).__init__()
        self.addon = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Sigmoid()
        )


def _addon_canonized(pretrained=False, progress=True, **kwargs):
    model = addon_canonized()
    return model


def generate_prp_all(dataloader, model, foldername, device):
    model.train(False)

    for pno in range(model.num_prototypes):
        i = 0
        for data in dataloader:
            # get the inputs
            inputs = data[0]
            fns = data[2]

            inputs = inputs.to(device).clone()

            inputs.requires_grad = True

            with torch.enable_grad():

                conv_features = model.conv_features(inputs)

                newl2 = l2_lrp_class.apply
                similarities = newl2(conv_features, model)

                # global max pooling
                min_distances = model.max_layer(similarities)

                min_distances = min_distances.view(-1, model.num_prototypes)

            '''For individual prototype'''
            (min_distances[:, pno]).backward()

            rel = inputs.grad.data
            print(fns)
            print("\n")
            #
            imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'), folder=foldername+str(pno)+"/", name=fns[0].split("/")[4])


def generate_prp_image(inputs, pno, model, device):
    model.train(False)
    inputs = inputs.to(device).clone()

    inputs.requires_grad = True

    with torch.enable_grad():
        conv_features = model.conv_features(inputs)

        newl2 = l2_lrp_class.apply
        similarities = newl2(conv_features, model)

        # global max pooling
        min_distances = model.max_layer(similarities)

        min_distances = min_distances.view(-1, model.num_prototypes)

    '''For individual prototype'''
    (min_distances[:, pno]).backward()

    rel = inputs.grad.data
    print("\n")
    #
    prp = imshow_im(rel.to('cpu'), imgtensor=inputs.to('cpu'))

    return prp


def heatmap(R, sx, sy, name):
    # b = 10*np.abs(R).mean()
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    plt.savefig(name)
    plt.clf()



def setbyname(obj, name, value):

    def iteratset(obj, components, value):

        if not hasattr(obj, components[0]):
            print(components[0])
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(obj, components, value)
    return success




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



def prp_canonize_model(ppnet,base_arch, device):

    model = base_architecture_to_features[base_arch](pretrained=False)
    model = model.to(device)


    lrp_params_def1 = {
        'conv2d_ignorebias': True,
        'eltwise_eps': 1e-6,
        'linear_eps': 1e-6,
        'pooling_eps': 1e-6,
        'use_zbeta': True,
    }

    lrp_layer2method = {
        'nn.ReLU': relu_wrapper_fct,
        'nn.Sigmoid': sigmoid_wrapper_fct,
        'nn.BatchNorm2d': relu_wrapper_fct,
        'nn.Conv2d': conv2d_beta0_wrapper_fct,
        'nn.Linear': linearlayer_eps_wrapper_fct,
        'nn.AdaptiveAvgPool2d': adaptiveavgpool2d_wrapper_fct,
        'nn.MaxPool2d': maxpool2d_wrapper_fct,
        'sum_stacked2': eltwisesum_stacked2_eps_wrapper_fct
    }

    model.copyfrom(ppnet.features, lrp_params=lrp_params_def1, lrp_layer2method=lrp_layer2method, verbose=False)
    model = model.to(device)
    ppnet.features = model

    conv_layer1 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer1.weight.data = ppnet.ones

    wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer1), lrp_params_def1, lrp_layer2method)
    conv_layer1 = wrapped

    conv_layer2 = nn.Conv2d(ppnet.prototype_shape[1], ppnet.prototype_shape[0], kernel_size=1, bias=False).to(device)
    conv_layer2.weight.data = ppnet.prototype_vectors

    wrapped = get_lrpwrapperformodule(copy.deepcopy(conv_layer2), lrp_params_def1, lrp_layer2method)
    conv_layer2 = wrapped

    relu_layer = nn.ReLU().to(device)
    wrapped = get_lrpwrapperformodule(copy.deepcopy(relu_layer), lrp_params_def1, lrp_layer2method)
    relu_layer = wrapped

    wrapped = get_lrpwrapperformodule(copy.deepcopy(ppnet.last_layer), lrp_params_def1, lrp_layer2method)
    last_layer = wrapped


    add_on_layers = _addon_canonized()
    for src_module_name, src_module in ppnet.add_on_layers.named_modules():
        if isinstance(src_module, nn.Conv2d):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.ReLU):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

        if isinstance(src_module, nn.Sigmoid):
            wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), lrp_params_def1, lrp_layer2method)
            setbyname(add_on_layers.addon, src_module_name, wrapped)

    ppnet.max_layer = torch.nn.MaxPool2d((7, 7), return_indices=False)

    ## Maxpool
    ppnet.max_layer = get_lrpwrapperformodule(copy.deepcopy(ppnet.max_layer), lrp_params_def1, lrp_layer2method)

    add_on_layers = add_on_layers.to(device)
    ppnet.add_on_layers = add_on_layers.addon

    ppnet.conv_layer1 = conv_layer1
    ppnet.conv_layer2 = conv_layer2
    ppnet.relu_layer = relu_layer
    ppnet.last_layer = last_layer
    ppnet.canonized = True
    return ppnet

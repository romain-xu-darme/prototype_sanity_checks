import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim
from typing import Tuple, List

"""
    Utility functions for handling parsed arguments

"""


def add_prototree_init_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add all options for the initialization of a ProtoTree """
    parser.add_argument('--net',
                        type=str,
                        metavar='<arch>',
                        default='resnet50_inat',
                        help='Base network used in the tree. Pretrained network on iNaturalist is only available '
                             'for resnet50_inat (default). Others are pretrained on ImageNet. '
                             'Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, '
                             'densenet121, densenet169, densenet201, densenet161, '
                             'vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or vgg19_bn')
    parser.add_argument('--depth',
                        type=int,
                        metavar='<num>',
                        default=9,
                        help='The tree is initialized as a complete tree of this depth')
    parser.add_argument('--W1',
                        type=int,
                        metavar='<width>',
                        default=1,
                        help='Width of the prototype. Correct behaviour of the model with W1 != 1 is not guaranteed')
    parser.add_argument('--H1',
                        type=int,
                        metavar='<height>',
                        default=1,
                        help='Height of the prototype. Correct behaviour of the model with H1 != 1 is not guaranteed')
    parser.add_argument('--num_features',
                        type=int,
                        metavar='<num>',
                        default=256,
                        help='Depth of the prototype and therefore also depth of convolutional output')
    parser.add_argument('--init_mode',
                        type=str,
                        metavar='<mode>',
                        default=None,
                        help='Either None, "pretrained", or path to a state dict file. \n'
                             '- None: the backbone network is initialized with random weights. \n'
                             '- "pretrained": resnet50_inat is initalized with weights from iNaturalist2017. Other '
                             'networks are initialized with weights from ImageNet.\n'
                             'Otherwise, both backbone and add-on layers are initialized with explicit state dict.')
    parser.add_argument('--disable_derivative_free_leaf_optim',
                        action='store_true',
                        help='Flag that optimizes the leafs with gradient descent when set instead of '
                             'using the derivative-free algorithm'
                        )
    parser.add_argument('--kontschieder_train',
                        action='store_true',
                        help='Flag that first trains the leaves for one epoch, and then trains the rest of ProtoTree '
                             '(instead of interleaving leaf and other updates). Computationally more expensive.'
                        )
    parser.add_argument('--kontschieder_normalization',
                        action='store_true',
                        help='Flag that disables softmax but uses a normalization factor to convert the '
                             'leaf parameters to a probabilitiy distribution, as done by Kontschieder et al. (2015). '
                             'Will iterate over the data 10 times to update the leaves. Computationally more expensive.'
                        )
    parser.add_argument('--focal_distance',
                        action='store_true',
                        help='Flag that enables the use of a focal distance, as described in Rymarczyk et al. (2021)'
                        )
    parser.add_argument('--log_probabilities',
                        action='store_true',
                        help='Flag that uses log probabilities when set. Useful when getting NaN values.'
                        )
    return parser


def add_ensemble_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add options for building a tree ensemble """
    parser.add_argument('--nr_trees_ensemble',
                        type=int,
                        metavar='<num>',
                        default=5,
                        help='Number of ProtoTrees to train and (optionally) use in an ensemble. '
                             'Used in main_ensemble.py')
    return parser


def add_general_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add general options that are used in most applications """
    parser.add_argument('--tree_dir',
                        type=str,
                        metavar='<path>',
                        default='',
                        help='The directory containing a state dict (checkpoint) with a pretrained prototree. ')
    parser.add_argument('--dataset',
                        type=str,
                        metavar='<name>',
                        default='CUB-200-2011',
                        help='Data set on which the ProtoTree should be trained')
    parser.add_argument('--batch_size',
                        type=int,
                        metavar='<num>',
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--device',
                        type=str,
                        metavar='<device>',
                        default='cuda:0',
                        help='Target device')
    parser.add_argument('--root_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Root directory where everything will be saved')
    parser.add_argument('--proj_dir',
                        type=str,
                        metavar='<path>',
                        default='projected',
                        help='Directoy for saving the prototypes, patches and heatmaps (inside root dir)')
    parser.add_argument('--upsample_threshold',
                        type=str,
                        metavar='<value>',
                        default="0.3",
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an '
                             'image after upsampling. The higher this threshold, the larger the patches. '
                             'If set to "auto", will use Otsu threshold instead.')
    parser.add_argument('--upsample_mode',
                        type=str,
                        metavar='<mode>',
                        default='vanilla',
                        choices=['vanilla', 'smoothgrads', 'prp'],
                        help='Upsampling mode. Either vanilla (cubic interpolation), smoothgrads or prp.')
    parser.add_argument('--grads_x_input',
                        action='store_true',
                        help='Flag that enables use of gradients x input for refined bounding boxes')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    return parser


def add_finalize_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--pruning_threshold_leaves',
                        type=float,
                        metavar='<threshold>',
                        default=0.01,
                        help='An internal node will be pruned when the maximum class probability in the distributions '
                             'of all leaves below this node are lower than this threshold.')
    parser.add_argument('--projection_mode',
                        type=str,
                        metavar='<mode>',
                        default='cropped',
                        choices=['raw', 'cropped', 'corners'],
                        help='Specify the preprocessing on the training set before projecting prototypes.'
                        )
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Add all options for the training of a ProtoTree """
    parser.add_argument('--epochs',
                        type=int,
                        metavar='<num>',
                        default=100,
                        help='The number of epochs the tree should be trained')
    parser.add_argument('--optimizer',
                        type=str,
                        metavar='<name>',
                        default='AdamW',
                        help='The optimizer that should be used when training the tree')
    parser.add_argument('--lr',
                        type=float,
                        metavar='<rate>',
                        default=0.001,
                        help='The optimizer learning rate for training the prototypes')
    parser.add_argument('--lr_block',
                        type=float,
                        metavar='<rate>',
                        default=0.001,
                        help='The optimizer learning rate for training the 1x1 conv layer and last conv layer '
                             'of the underlying neural network (applicable to resnet50 and densenet121)')
    parser.add_argument('--lr_net',
                        type=float,
                        metavar='<rate>',
                        default=1e-5,
                        help='The optimizer learning rate for the underlying neural network')
    parser.add_argument('--lr_pi',
                        type=float,
                        metavar='<rate>',
                        default=0.001,
                        help='The optimizer learning rate for the leaf distributions '
                             '(only used if disable_derivative_free_leaf_optim flag is set')
    parser.add_argument('--momentum',
                        type=float,
                        metavar='<value>',
                        default=0.9,
                        help='The optimizer momentum parameter (only applicable to SGD)')
    parser.add_argument('--weight_decay',
                        type=float,
                        metavar='<value>',
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--milestones',
                        type=str,
                        metavar='<value>',
                        default='',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        metavar='<value>',
                        default=0.5,
                        help='The gamma for the MultiStepLR learning rate scheduler. Needs to be 0<=gamma<=1')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        metavar='<num>',
                        default=2,
                        help='Number of epochs where pretrained features_net will be frozen'
                        )
    parser.add_argument('--skip_eval_after_training',
                        action='store_true',
                        help='Skip network evaluation after pruning and projection.'
                        )
    parser.add_argument('--force',
                        action='store_true',
                        help='Overwrite output directory when it exists.'
                        )
    parser = add_finalize_args(parser)
    return parser


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    if hasattr(args, 'random_seed'):
        # Init random seeds
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
    if hasattr(args, 'milestones'):
        args.milestones = get_milestones(args)
    return args


def get_milestones(args: argparse.Namespace):
    """ Parse the milestones argument to get a list

        :param args: The arguments given
    """
    if args.milestones != '':
        milestones_list = args.milestones.split(',')
        for m in range(len(milestones_list)):
            milestones_list[m] = int(milestones_list[m])
    else:
        milestones_list = []
    return milestones_list


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exist, create it
    os.makedirs(directory_path, exist_ok=True)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)


def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args


def get_optimizer(tree, args: argparse.Namespace) -> Tuple[torch.optim.Optimizer, List, List]:
    """
    Construct the optimizer as dictated by the parsed arguments

    :param tree: The tree that should be optimized
    :param args: Parsed arguments containing hyperparameters. The '--optimizer' argument specifies which type of
                 optimizer will be used. Optimizer specific arguments (such as learning rate and momentum) can be passed
                 this way as well
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen,
        and parameter set of the net that will be trained
    """

    optim_type = args.optimizer
    # create parameter groups
    params_to_freeze = []
    params_to_train = []

    dist_params = []
    for name, param in tree.named_parameters():
        if 'dist_params' in name:
            dist_params.append(param)
    # set up optimizer
    if 'resnet50_inat' in args.net or ('resnet50' in args.net and args.dataset == 'CARS'):
        # to reproduce experimental results
        # freeze resnet50 except last convolutional layer
        for name, param in tree._net.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        if optim_type == 'SGD':
            paramlist = [
                {"params": params_to_freeze,
                    "lr": args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": params_to_train,
                    "lr": args.lr_block, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": tree._add_on.parameters(),
                    "lr": args.lr_block, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": tree.prototype_layer.parameters(),
                    "lr": args.lr, "weight_decay_rate": 0, "momentum": 0}]
            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})
        else:
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]

            if args.disable_derivative_free_leaf_optim:
                paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})

    else:  # other network architectures
        for name, param in tree._net.named_parameters():
            params_to_freeze.append(param)
        paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": tree._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": tree.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
        if args.disable_derivative_free_leaf_optim:
            paramlist.append({"params": dist_params, "lr": args.lr_pi, "weight_decay_rate": 0})

    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist,
                               lr=args.lr,
                               momentum=args.momentum), params_to_freeze, params_to_train
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist, lr=args.lr, eps=1e-07), params_to_freeze, params_to_train
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist, lr=args.lr, eps=1e-07, weight_decay=args.weight_decay), \
               params_to_freeze, params_to_train

    raise Exception('Unknown optimizer argument given!')

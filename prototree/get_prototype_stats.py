import copy
import os

import numpy as np
from tqdm import tqdm
from util.args import *
from typing import List, Callable
from util.data import get_dataloaders
from prototree.prototree import ProtoTree
from PIL import Image
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from prototree.upsample import find_threshold_to_area, find_mask_to_area, convert_bbox_coordinates
from util.gradients import smoothgrads, prp, cubic_upsampling, randgrads, normalize_min_max
from features.prp import canonize_tree
import torch
import cv2

# Use only deterministic algorithms
torch.use_deterministic_algorithms(True)

supported_methods = {
    'smoothgrads': smoothgrads,
    'prp': prp,
    'vanilla': cubic_upsampling,
    'randgrads': randgrads
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Compare different upsampling modes when upsampling prototypes')
    parser.add_argument('--tree_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='The directory containing a state dict (checkpoint) with a pretrained prototree. ')
    parser.add_argument('--base_arch',
                        type=str,
                        metavar='<arch>',
                        required=True,
                        help='Architecture of feature extractor (for PRP).')
    parser.add_argument('--dataset',
                        type=str,
                        metavar='<name>',
                        required=True,
                        help='Data set on which the ProtoTree has been trained')
    parser.add_argument('--methods',
                        type=str, nargs='+',
                        metavar='<name>',
                        default=['smoothgrads', 'prp', 'vanilla', 'randgrads'],
                        help='List of saliency methods')
    parser.add_argument('--use-segmentation',
                        action='store_true',
                        help='Use segmentation')
    parser.add_argument('--mode',
                        type=str,
                        choices=['bbox', 'mask'],
                        metavar='<mode>',
                        default='mask',
                        help='Mode of extraction (either bounding box or more precise mask)')
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
    parser.add_argument('--proj_dir',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Directory for saving the prototypes visualizations')
    parser.add_argument('--output',
                        type=str,
                        metavar='<name>',
                        required=True,
                        help='Stats file name inside projection directory.')
    parser.add_argument('--target_areas',
                        type=float,
                        nargs=3,
                        metavar=('<min>', '<max>', '<increment>'),
                        help='Target bounding box areas')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Do not save debug images')
    add_finalize_args(parser)
    return parser


def compute_prototype_stats(
        tree: ProtoTree,
        canonized_tree: ProtoTree,
        img: Image,
        segm: Image,
        img_tensor: torch.Tensor,
        node_id: int,
        methods: List[str],
        depth: int,
        transform: Callable,
        img_name: str,
        target_areas: List[float],
        mode: str,
        output_dir: str,
        output_filename: str = 'prototype_stats.csv',
        location: Tuple[int, int] = None,
        device: str = 'cuda:0',
        quiet: bool = False,
) -> None:
    """ Compute fidelity and relevance stats for a given prototype

    :param tree: ProtoTree
    :param canonized_tree: Modified tree for PRP
    :param img: Original image
    :param segm: Image segmentation (if any)
    :param img_tensor: Image tensor
    :param node_id: Node ID = index of the prototype in the similarity map
    :param methods: List of saliency methods
    :param depth: Node depth inside the tree
    :param transform: Preprocessing function
    :param img_name: Will be used in the statistic file
    :param target_areas: Target bounding box areas
    :param mode: Either bounding box or mask
    :param output_dir: Destination folder
    :param output_filename: File name
    :param location: These coordinates are used to determine the upsampling target location
    :param device: Target device
    :param quiet: In quiet mode, does not create images with bounding boxes
    """
    segm = segm if segm is None else np.asarray(segm)

    for method in methods:
        func = supported_methods[method]
        # Compute gradients
        grads = func(
            tree=tree if method != prp else canonized_tree,
            img_tensor=copy.deepcopy(img_tensor),
            node_id=node_id,
            location=location,
            device=device,
        )
        grads = cv2.resize(grads, dsize=(img.width, img.height), interpolation=cv2.INTER_CUBIC)
        grads = normalize_min_max(grads)
        if mode == "mask":
            # Sort gradients once
            grads_sorted = np.sort(np.reshape(grads, (-1)))

        img_tensors = [img_tensor.clone().detach()]
        relevances = []
        for target_area in target_areas:
            if mode == 'bbox':
                xmin, xmax, ymin, ymax, _ = find_threshold_to_area(grads, target_area)

                # Measure intersection with segmentation (if provided)
                relevance = np.sum(np.sum(segm[ymin:ymax, xmin:xmax], axis=2) > 0) if segm is not None else 0
                relevance /= ((ymax - ymin) * (xmax - xmin))+1e-14
                relevances.append(relevance)

                # Accumulate perturbed images (will be processed in batch)
                # WARNING: bounding box coordinates have been computed on original image dimension, we need to convert them
                xmin_r, xmax_r, ymin_r, ymax_r = convert_bbox_coordinates(
                    xmin, xmax, ymin, ymax,
                    img.width, img.height,
                    img_tensor.size(2), img_tensor.size(3),
                )
                deleted_img = img_tensor.clone().detach()
                deleted_img[0, :, ymin_r:ymax_r, xmin_r:xmax_r] = 0
                img_tensors.append(deleted_img)
            else:
                # Get binary mask of most salient pixels
                mask, _ = find_mask_to_area(grads, grads_sorted, target_area)

                # Measure intersection with segmentation (if provided)
                relevance = np.sum((segm[:, :, 0] > 0) * mask) * 1.0 if segm is not None else 0.0
                relevance /= (np.sum(mask) + 1e-14)
                relevances.append(relevance)

                # Accumulate perturbed images (will be processed in batch)
                mask = np.expand_dims(mask, 2)
                # Set most salient pixels to mean value
                mean = transform.transforms[2].mean
                img_array = np.uint8(np.asarray(img) * (1 - mask) + mask * mean * 255)
                img_tensors.append(transform(Image.fromarray(img_array)).unsqueeze(0).to(img_tensor.device))

        # Compute fidelities
        img_tensors = torch.cat(img_tensors, dim=0)
        with torch.no_grad():
            _, distances_batch, _ = tree.forward_partial(img_tensors)
            sim_map = torch.exp(-distances_batch[:, node_id]).cpu().numpy()
        h, w = location
        ref_similarity = sim_map[0, h, w]
        fidelities = sim_map[1:, h, w] / ref_similarity

        # Compute intersection betwen 98 percentile mask and segmentation
        mask = grads > np.percentile(grads, 98)
        relevance_98pc = np.sum((segm[:, :, 0] > 0) * mask) * 1.0 / (np.sum(mask) + 1e-14) if segm is not None else 0.0

        with open(os.path.join(output_dir, output_filename), 'a') as fout:
            for area, relevance, fidelity in zip(target_areas, relevances, fidelities):
                fout.write(f'{img_name},{node_id},{depth},'
                           f'{method},{area},{relevance},{fidelity},{relevance_98pc}\n')

        if not quiet:
            mask = np.expand_dims(mask, 2)
            saved_image = np.uint8(np.asarray(img) * (1 - mask) + mask * (255, 255, 0))
            Image.fromarray(saved_image).save(f'{output_dir}/{img_name}_{node_id}_{method}.png')


def finalize_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args

    # Obtain the dataset and dataloaders
    _, projectloader, _, _, _ = get_dataloaders(
        dataset=args.dataset,
        projection_mode=args.projection_mode,
        batch_size=args.batch_size,
        device=args.device,
    )
    # Recover preprocessing function
    transform = projectloader.dataset.transform
    target_areas = np.arange(args.target_areas[0], args.target_areas[1] + args.target_areas[2], args.target_areas[2])
    os.makedirs(args.proj_dir, exist_ok=True)

    # Reset stat file
    with open(os.path.join(args.proj_dir, args.output), 'w') as fout:
        fout.write('image name,node id,depth,method,area,relevance,fidelity,non biased\n')

    # Load tree
    tree = ProtoTree.load(args.tree_dir, map_location=args.device)

    # Indicate backbone architecture for PRP canonization
    tree.base_arch = args.base_arch

    # Pruning
    prune(tree, args.pruning_threshold_leaves, None)

    # Projection
    project_info, tree = project_with_class_constraints(tree, projectloader, args.device, None)
    tree.eval()

    # Use canonized tree for PRP
    canonized_tree = canonize_tree(copy.deepcopy(tree), arch=tree.base_arch, device=args.device)

    segm_dir = None
    if args.use_segmentation:
        assert args.dataset.startswith('CUB'), "Segmentation mask only supported for CUB200"
        if args.projection_mode == 'raw':
            segm_dir = "../data/CUB_200_2011/dataset/train_full_seg"
        elif args.projection_mode == "corners":
            segm_dir = "../data/CUB_200_2011/dataset/train_corners_seg"
        else:
            segm_dir = "../data/CUB_200_2011/dataset/train_crop_seg"

    # Raw images from projection set
    imgs = projectloader.dataset.imgs
    for node, j in tqdm(tree._out_map.items()):
        if node in tree.branches:  # do not upsample when node is pruned
            prototype_info = project_info[j]
            node_name = prototype_info['node_ix']
            # Open image without preprocessing
            img_path = imgs[prototype_info['input_image_ix']][0]
            x = Image.open(img_path).convert('RGB')
            fname = os.path.splitext(os.path.basename(img_path))[0]

            segm = None
            if segm_dir:
                target_dir = os.path.join(segm_dir, img_path.split('/')[-2])
                assert os.path.isfile(os.path.join(target_dir, fname + '.jpg')) or \
                       os.path.isfile(os.path.join(target_dir, fname + '.png')), f"Segmentation not found for {fname}"
                ext = '.jpg' if os.path.isfile(os.path.join(target_dir, fname + '.jpg')) else '.png'
                segm = Image.open(os.path.join(target_dir, fname + ext)).convert('RGB')

            prototype_location = prototype_info['patch_ix']
            W, H = prototype_info['W'], prototype_info['H']

            compute_prototype_stats(
                tree=tree,
                canonized_tree=canonized_tree,
                img=x,
                segm=segm,
                img_tensor=prototype_info['nearest_input'],
                node_id=tree._out_map[node],
                methods=args.methods,
                depth=len(tree.path_to(node)),
                transform=transform,
                img_name=f'proto_{node_name}',
                output_dir=args.proj_dir,
                output_filename=args.output,
                target_areas=list(target_areas),
                mode=args.mode,
                location=(prototype_location // H, prototype_location % H),
                device=args.device,
                quiet=args.quiet,
            )


if __name__ == '__main__':
    finalize_tree()

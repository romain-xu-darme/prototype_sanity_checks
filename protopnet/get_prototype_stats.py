import torch.nn as nn
import copy
import os
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Callable, Tuple
from PIL import Image
from gradients import smoothgrads, prp, cubic_upsampling, randgrads, normalize_min_max
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import push
from prp.prp import prp_canonize_model
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


def find_high_activation_crop(mask, threshold):
    threshold = 1.-threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def find_mask_to_area(
        grads: np.array,
        sorted_grads: np.array,
        area: float,
) -> Tuple[np.array, float]:
    """ Given a gradient heatmap and a target area, determine the optimum threshold to achieve a mask
    with desired area

    :param grads: Heatmap (normalized between 0 and 1)
    :param sorted_grads: Sorted grads (if possible)
    :param area: Desired area
    :returns: Mask and effective area
    """
    sorted_grads = np.sort(np.reshape(grads, (-1))) if sorted_grads is None else sorted_grads
    threshold = sorted_grads[int(len(sorted_grads)*(1-area))]
    mask = (grads > threshold)
    return mask, 1.0*np.sum(mask)/(mask.shape[0]*mask.shape[1])


def find_threshold_to_area(
        grads: np.array,
        area: float,
        precision: float = 0.01,
        ntries: int = 30
) -> Tuple[int, int, int, int, float]:
    """ Given a gradient heatmap and a target area, determine the optimum threshold to achieve a bounding box
    with desired area, within a given number of tries

    :param grads: Heatmap (normalized between 0 and 1)
    :param area: Desired area
    :param precision: Area precision
    :param ntries: Number of attempts
    :returns: Bounding box parameters (xmin, xmax, ymin, ymax) and effective area
    """
    # Find optimal threshold through dichotomic search
    tmin = 0.0
    tmax = 1.0
    xmin = xmax = ymin = ymax = area_ratio = 0
    for itry in range(ntries):
        tcur = (tmax + tmin) / 2
        # Compute corresponding bounding box
        high_act_patch_indices = find_high_activation_crop(grads, tcur)
        ymin, ymax = high_act_patch_indices[0], high_act_patch_indices[1]
        xmin, xmax = high_act_patch_indices[2], high_act_patch_indices[3]
        bbox_area = ((ymax - ymin) * (xmax - xmin))
        img_area = grads.shape[0] * grads.shape[1] * 1.0
        area_ratio = float(bbox_area / img_area)
        if abs(area_ratio - area) <= precision:
            # Found acceptable bounding box
            break
        else:
            if area_ratio > area:
                tmax = tcur
            else:
                tmin = tcur
    return xmin, xmax, ymin, ymax, area_ratio


def convert_bbox_coordinates(
        xmin: int, xmax: int, ymin: int, ymax: int,
        src_width: int, src_height: int,
        dst_width: int, dst_height: int
) -> Tuple[int, int, int, int]:
    """ Convert bounding box coordinates from one image dimension to another """
    xmin_resized = int(xmin * dst_width / src_width)
    xmax_resized = int(xmax * dst_width / src_width)
    ymin_resized = int(ymin * dst_height / src_height)
    ymax_resized = int(ymax * dst_height / src_height)
    return xmin_resized, xmax_resized, ymin_resized, ymax_resized


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Given a trained ProtoPNet, '
                                     'evaluate the average fidelity and relevance score of prototypes')
    parser.add_argument('--model',
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
                        help='Data set on which the ProtoPNet has been trained')
    parser.add_argument('--segm_dir',
                        type=str,
                        metavar='<path>',
                        help='Directory to segmentation of train images (if available)')
    parser.add_argument('--mode',
                        type=str,
                        choices=['bbox', 'mask'],
                        metavar='<mode>',
                        default='mask',
                        help='Mode of extraction (either bounding box or more precise mask)')
    parser.add_argument('--methods',
                        type=str, nargs='+',
                        metavar='<name>',
                        default=['smoothgrads', 'prp', 'vanilla', 'randgrads'],
                        help='List of saliency methods')
    parser.add_argument('--batch_size',
                        type=int,
                        metavar='<num>',
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--proj_dir',
                        type=str,
                        metavar='<path>',
                        help='Path to projection directory')
    parser.add_argument('--device',
                        type=str,
                        metavar='<device>',
                        default='cuda:0',
                        help='Target device')
    parser.add_argument('--output',
                        type=str,
                        metavar='<path>',
                        required=True,
                        help='Path to stats file.')
    parser.add_argument('--target_areas',
                        type=float,
                        nargs=3,
                        metavar=('<min>', '<max>', '<increment>'),
                        help='Target bounding box areas')
    parser.add_argument('--img_size',
                        type=int,
                        metavar='<size>',
                        default=224,
                        help='Image size')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Quiet mode')
    parsed_args = parser.parse_args()
    return parsed_args


def compute_prototype_stats(
        ppnet: nn.Module,
        canonized_ppnet: nn.Module,
        img: Image,
        segm: Image,
        img_tensor: torch.Tensor,
        proto_id: int,
        methods: List[str],
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

    :param ppnet: ProtoPNet
    :param canonized_ppnet: Modified model for PRP
    :param img: Original image
    :param segm: Image segmentation (if any)
    :param img_tensor: Image tensor
    :param proto_id: Index of the prototype in the similarity map
    :param methods: List of saliency methods
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
        func  =supported_methods[method]
        # Compute gradients
        grads = func(
            ppnet=ppnet if method != prp else canonized_ppnet,
            img_tensor=copy.deepcopy(img_tensor),
            proto_id=proto_id,
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
                if segm is not None:
                    relevance = np.sum((segm[:, :, 0] > 0) * mask) * 1.0
                    relevance /= np.sum(mask) + 1e-14
                else:
                    relevance = 0.0
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
            _, distances = ppnet.push_forward(img_tensors)  # Shapes _ , N x P x H x W
            # Convert distances to similarities
            prototype_activation_patterns = ppnet.distance_2_similarity(distances).detach().cpu().numpy()
        if ppnet.prototype_activation_function == 'linear':
            prototype_shape = ppnet.prototype_shape
            max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
            prototype_activation_patterns = prototype_activation_patterns + max_dist
        # Find reference location of most similar patch
        h, w = np.where(
            prototype_activation_patterns[0, proto_id] == np.max(prototype_activation_patterns[0, proto_id]))
        h, w = h[0], w[0]
        ref_similarity = prototype_activation_patterns[0, proto_id, h, w]
        fidelities = prototype_activation_patterns[1:, proto_id, h, w] / ref_similarity

        # Compute intersection betwen 98 percentile mask and segmentation
        mask = grads > np.percentile(grads, 98)
        relevance_98pc = np.sum((segm[:, :, 0] > 0) * mask) * 1.0 / (np.sum(mask) + 1e-14) if segm is not None else 0.0

        with open(os.path.join(output_dir, output_filename), 'a') as fout:
            for area, relevance, fidelity in zip(target_areas, relevances, fidelities):
                fout.write(f'{img_name},{proto_id},{method},{area},{relevance},{fidelity},{relevance_98pc}\n')

        if not quiet:
            mask = np.expand_dims(mask, 2)
            saved_image = np.uint8(np.asarray(img) * (1 - mask) + mask * (255, 255, 0))
            Image.fromarray(saved_image).save(f'{output_dir}/{proto_id}_{img_name}_{method}.png')


def finalize():
    args = get_args()

    # Load model
    ppnet = torch.load(args.model, map_location=args.device)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    # Prepare projection data
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    img_size = ppnet.img_size
    transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
    ])
    projectloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.dataset, transform=transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    protos_infos = push.get_prototype_infos(
        ppnet_parallel=ppnet_multi,
        dataloader=projectloader)

    target_areas = np.arange(args.target_areas[0], args.target_areas[1] + args.target_areas[2], args.target_areas[2])

    # Reset stat file
    os.makedirs(args.proj_dir, exist_ok=True)
    with open(os.path.join(args.proj_dir, args.output), 'w') as fout:
        fout.write('image name,node id,method,area,relevance,fidelity,non biased\n')

    ppnet.eval()
    # Use canonized net for PRP
    canonized_ppnet = prp_canonize_model(copy.deepcopy(ppnet), base_arch=args.base_arch, device=args.device)

    for prototype_info in tqdm(protos_infos):
        # Open image without preprocessing
        img_path = prototype_info['path']
        x = Image.open(img_path).convert('RGB')
        fname = os.path.splitext(os.path.basename(img_path))[0]
        segm = Image.open(os.path.join(args.segm_dir, img_path.split('/')[-2], fname + '.jpg')).convert('RGB') \
            if args.segm_dir is not None else None

        compute_prototype_stats(
            ppnet=ppnet,
            canonized_ppnet=canonized_ppnet,
            img=x,
            segm=segm,
            img_tensor=transform(x).unsqueeze(0).to(args.device),
            proto_id=prototype_info['id'],
            methods=args.methods,
            transform=transform,
            img_name=fname,
            output_dir=args.proj_dir,
            output_filename=args.output,
            target_areas=list(target_areas),
            mode=args.mode,
            location=prototype_info['location'],
            device=args.device,
            quiet=args.quiet,
        )


if __name__ == '__main__':
    finalize()

import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
from typing import List, Union, Tuple, Callable
from PIL import Image
import argparse
import torch
import random
from tqdm import tqdm
import os
import copy
import cv2
import numpy as np
from gradients import smoothgrads, prp, cubic_upsampling, randgrads, normalize_min_max
from prp.prp import prp_canonize_model

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
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


def compute_inference_stats(
        ppnet: nn.Module,
        canonized_ppnet: nn.Module,
        img: Image.Image,
        segm: Union[Image.Image, None],
        img_tensor: torch.Tensor,
        transform: Callable,
        label: int,
        methods: List[str],
        mode: str,
        img_name: str,
        target_areas: List[float],
        output: str,
        device: str,
) -> None:
    """ Compute fidelity and relevance statistics for a given image

    :param ppnet: ProtoPNet
    :param canonized_ppnet: Modified model for PRP
    :param img: Original image
    :param segm: Image segmentation (if any)
    :param img_tensor: Input image tensor
    :param transform: Preprocessing function (for generating masked images)
    :param label: Ground truth label
    :param methods: List of saliency methods
    :param mode: Either bounding box or mask
    :param img_name: Image name
    :param target_areas: Find threshold such that bounding boxes cover a given area
    :param output: Path to output file
    :param device: Target device
    :returns: Prediction, fidelity statistics for each node used in positive reasoning
    """
    segm = segm if segm is None else np.asarray(segm)

    # Get the model prediction
    with torch.no_grad():
        pred, min_distances = ppnet(img_tensor)  # Shapes N x C, N x P
        label_ix = torch.argmax(pred, dim=1)[0].item()
        prototype_activations = ppnet.distance_2_similarity(min_distances)
        if ppnet.prototype_activation_function == 'linear':
            prototype_shape = ppnet.prototype_shape
            max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
            prototype_activations = prototype_activations + max_dist
        _, sorted_indices_act = torch.sort(prototype_activations[0])  # Shape P

    # Keep 10 most activated prototypes
    most_highly_activated_protos = [sorted_indices_act[-i].item() for i in range(1, 11)]

    for rank, proto_id in enumerate(most_highly_activated_protos):
        # Vanilla (white), PRP (purple) and Smoothgrads (yellow)
        for method in methods:
            func = supported_methods[method]
            # Compute gradients
            grads = func(
                ppnet=ppnet if method != prp else canonized_ppnet,
                img_tensor=copy.deepcopy(img_tensor),
                proto_id=proto_id,
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
                        relevance = np.sum((segm[:, :, 0]>0)*mask)*1.0
                        relevance /= np.sum(mask)+1e-14
                    else:
                        relevance = 0.0
                    relevances.append(relevance)

                    # Accumulate perturbed images (will be processed in batch)
                    mask = np.expand_dims(mask, 2)
                    # Set most salient pixels to mean value
                    img_array = np.uint8(np.asarray(img) * (1-mask) + mask * mean * 255)
                    img_tensors.append(transform(Image.fromarray(img_array)).unsqueeze(0).to(img_tensor.device))

            # Compute fidelities
            img_tensors = torch.cat(img_tensors, dim=0)
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

            with open(output, 'a') as fout:
                for area, relevance, fidelity in zip(target_areas, relevances, fidelities):
                    fout.write(f'{img_name}, {label}, {int(label_ix)}, {proto_id}, {rank}, '
                               f'{method}, {area:.3f}, {relevance:.3f}, {fidelity:.3f}\n')


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Given a dataset of test images, '
                                     'evaluate the average fidelity and relevance score')
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
                        help='Data set on which the ProtoTree has been trained')
    parser.add_argument('--segm_dir',
                        type=str,
                        metavar='<path>',
                        help='Directory to segmentation of train images (if available)')
    parser.add_argument('--mode',
                        type=str,
                        choices=['bbox', 'mask'],
                        metavar='<mode>',
                        default='bbox',
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
                        metavar=('<min>','<max>','<increment>'),
                        help='Target bounding box areas')
    parser.add_argument('--img_size',
                        type=int,
                        metavar='<size>',
                        default=224,
                        help='Image size')
    parser.add_argument('--restart_from',
                        type=str,
                        metavar='<name>',
                        help='Restart from a given image name')
    parser.add_argument('--random_seed',
                        type=int,
                        metavar='<seed>',
                        default=0,
                        help='Random seed (for reproducibility)')
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = get_args()
    # Init random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Prepare preprocessing
    normalize = transforms.Normalize(mean=mean, std=std)
    transform = transforms.Compose([
        transforms.Resize(size=(args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    img_set = torchvision.datasets.ImageFolder(args.dataset, transform=None)

    # Load ProtoPNet
    ppnet = torch.load(args.model, map_location=args.device)
    # Indicate backbone architecture for PRP canonization
    ppnet.eval()
    # Use canonized tree for PRP
    canonized_ppnet = prp_canonize_model(copy.deepcopy(ppnet), base_arch=args.base_arch, device=args.device)

    if not os.path.exists(args.output):
        with open(args.output, 'w') as f:
            f.write('path, label, pred, node id, rank, method, area, relevance, fidelity\n')

    wait = args.restart_from is not None
    target_areas = np.arange(args.target_areas[0], args.target_areas[1] + args.target_areas[2], args.target_areas[2])

    stats_iter = tqdm(
        enumerate(img_set),
        total=len(img_set),
        desc='Computing fidelity stats')
    for index, (img, label) in stats_iter:
        img_path = img_set.samples[index][0]
        # Raw file name
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        if wait and img_name != args.restart_from:
            continue
        wait = False
        segm = Image.open(os.path.join(args.segm_dir, img_path.split('/')[-2], img_name + '.jpg')).convert('RGB') \
            if args.segm_dir is not None else None
        compute_inference_stats(
            ppnet=ppnet,
            canonized_ppnet=canonized_ppnet,
            img=img,
            segm=segm,
            img_tensor=transform(img).unsqueeze(0).to(args.device),
            transform=transform,
            label=label,
            methods=args.methods,
            mode=args.mode,
            img_name=img_name,
            target_areas=target_areas,
            output=args.output,
            device=args.device,
        )

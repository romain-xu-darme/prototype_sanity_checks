import numpy as np
import argparse
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from gradients import cubic_upsampling, smoothgrads, prp
from helpers import find_high_activation_crop
import os


def get_local_expl_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--ppnet',
                        type=str,
                        metavar='<path>',
                        help='Path to ProtoPNet model')
    parser.add_argument('--proj-dir',
                        type=str,
                        metavar='<path>',
                        help='Path to projection directory')
    parser.add_argument('--image',
                        type=str,
                        metavar='<path>',
                        help='Path to image to be explained')
    parser.add_argument('--mode',
                        type=str,
                        metavar='<mode>',
                        choices=['vanilla', 'smoothgrads', 'prp'],
                        default='vanilla',
                        help='Visualization mode')
    parser.add_argument('--results_dir',
                        type=str,
                        metavar='<path>',
                        default='local_explanations',
                        help='Directory where local explanations will be saved')
    parser.add_argument('--seg_dir',
                        type=str,
                        metavar='<path>',
                        help='Directory to segmentation of images to be explained')
    parser.add_argument('--device',
                        type=str,
                        metavar='<name>',
                        default='cuda:0',
                        help='Device')
    parsed_args = parser.parse_args()
    return parsed_args



def save_prototype(fname, proj_dir, index):
    p_img = plt.imread(os.path.join(proj_dir, 'prototype-img' + str(index) + '.png'))
    plt.imsave(fname, p_img)


def save_prototype_self_activation(fname, proj_dir, index):
    p_img = plt.imread(os.path.join(proj_dir,
                                    'prototype-img-original_with_self_act' + str(index) + '.png'))
    plt.imsave(fname, p_img)


def save_prototype_original_img_with_bbox(fname, proj_dir, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(
        os.path.join(proj_dir, 'prototype-img-original' + str(index) + '.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


def ppnet_inference(
        ppnet: nn.Module,
        img_tensor: torch.Tensor,

):
    logits, min_distances = ppnet(img_tensor)  # Shapes N x C, N x P
    conv_output, distances = ppnet.push_forward(img_tensor)  # Shapes N x D x H x W, N x P x H x W
    # Convert distances to similarities
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    predicted_cls = torch.argmax(logits, dim=1)[0].item()
    print('Predicted: ' + str(predicted_cls))

    print('Most activated 10 prototypes of this image:')
    # Prototypes are sorted from least to most activated
    array_act, sorted_indices_act = torch.sort(prototype_activations[0])  # Shape P, P

    return predicted_cls, [sorted_indices_act[-i].item() for i in range(1, 11)]


if __name__ == '__main__':
    args = get_local_expl_args()

    # Load model
    ppnet = torch.load(args.ppnet, map_location=args.device)
    img_size = ppnet.img_size

    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
    prototype_info = np.load(args.proj_dir)
    prototype_img_identity = prototype_info[:, -1]
    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        print('All prototypes connect most strongly to their respective classes.')
    else:
        print('WARNING: Not all prototypes connect most strongly to their respective classes.')

    print('Device used: ', args.device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    model_base_architecture = args.ppnet.split('/')[-3]

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'most_activated_prototypes'), exist_ok=True)

    img_tensor = test_transform(Image.open(args.image)).unsqueeze(0).to(args.device)
    cls, proto_idxs = ppnet_inference(ppnet, img_tensor)
    for i, proto_idx in enumerate(proto_idxs):
        if args.mode == 'vanilla':
            grads = cubic_upsampling(ppnet, img_tensor, proto_idx)
        elif args.mode == 'smoothgrads':
            grads = smoothgrads(
                ppnet=ppnet,
                img_tensor=img_tensor,
                proto_id=proto_idx,
                location=None,
                device=args.device,
            )
        elif args.mode == 'prp':
            grads = prp(
                ppnet=ppnet,
                base_arch=model_base_architecture,
                img_tensor=img_tensor,
                proto_id=proto_idx,
                location=None,
                device=args.device,
            )
        upsampled_activation_pattern = cv2.resize(grads, dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC)

        # show the most highly activated patch of the image by this prototype
        original_img = cv2.resize(np.asarray(Image.open(args.image)), dsize=(img_size, img_size),
                                    interpolation=cv2.INTER_CUBIC)/255.0
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.imsave(os.path.join(args.results_dir, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                   high_act_patch)
        print('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(args.results_dir, 'most_activated_prototypes',
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        print('prototype activation map of the chosen image:')
        plt.imsave(os.path.join(args.results_dir, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                   overlayed_img)

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional
from prp.prp import prp_canonize_model
from prp.lrp_general6 import l2_lrp_class

def polarity_and_collapse(
        array: np.array,
        polarity: Optional[str] = None,
        avg_chan: Optional[int] = None,
) -> np.array:
    """ Apply polarity filter (optional) followed by average over channels (optional)

    :param array: Target
    :param polarity: Polarity (positive, negative, absolute)
    :param avg_chan: Dimension across which channels are averaged
    """
    assert polarity in [None, 'positive', 'negative', 'absolute'], f'Invalid polarity {polarity}'

    # Polarity first
    if polarity == 'positive':
        array = np.maximum(0, array)
    elif polarity == 'negative':
        array = np.abs(np.minimum(0, array))
    elif polarity == 'absolute':
        array = np.abs(array)

    # Channel average
    if avg_chan is not None:
        array = np.average(array, axis=avg_chan)
    return array


def normalize_min_max(array: np.array) -> np.array:
    """ Perform min-max normalization of a numpy array

    :param array: Target
    """
    vmin = np.amin(array)
    vmax = np.amax(array)
    # Avoid division by zero
    return (array - vmin) / (vmax - vmin + np.finfo(np.float32).eps)


def cubic_upsampling(
        ppnet: nn.Module,
        img_tensor: torch.Tensor,
        proto_id: int,
        location: Tuple[int, int] = None,
        device: str = None,
) -> np.array:
    # Partial passes
    _, distances = ppnet.push_forward(img_tensor)  # Shapes N x D x H x W, N x P x H x W

    # Convert distances to similarities
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)[0]
    if ppnet.prototype_activation_function == 'linear':
        prototype_shape = ppnet.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    return prototype_activation_patterns[proto_id].detach().cpu().numpy()


def smoothgrads(
        ppnet: nn.Module,
        img_tensor: torch.Tensor,
        proto_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
        nsamples: Optional[int] = 10,
        noise: Optional[float] = 0.2,
        grad_x_input: Optional[bool] = True,
) -> np.array:
    """ Perform patch visualization using SmoothGrad

    :param ppnet: ProtoPnet
    :param img_tensor: Input image tensor
    :param proto_id: Prototype index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param nsamples: Number of samples
    :param noise: Noise level
    :param grad_x_input: Multiply gradient by input
    :return: gradient map
    """
    if location is None:
        # Find location of feature vector closest to target node
        with torch.no_grad():
            _, distances_batch = ppnet.push_forward(img_tensor)  # Shapes N x P x H x W
        distances_batch = distances_batch[0, proto_id].cpu().numpy()  # Shape H x W
        (h, w) = np.where(distances_batch == np.min(distances_batch))
        h, w = h[0], w[0]
    else:
        # Location is predefined
        h, w = location

    if nsamples == 1:
        noisy_images = [img_tensor]
    else:
        # Compute variance from noise ratio
        sigma = (img_tensor.max() - img_tensor.min()).cpu().numpy() * noise
        # Generate noisy images around original.
        noisy_images = [img_tensor + torch.randn(img_tensor.shape).to(device) * sigma for _ in range(nsamples)]

    # Compute gradients
    grads = []
    for x in noisy_images:
        x.requires_grad_()
        # Forward pass
        _, distances_batch = ppnet.push_forward(x)
        # Identify target location before backward pass
        output = distances_batch[0, proto_id, h, w]
        output.backward(retain_graph=True)
        grads.append(x.grad.data[0].detach().cpu().numpy())

    # grads has shape (nsamples) x img_tensor.shape => average across all samples
    grads = np.mean(np.array(grads), axis=0)
    if grad_x_input:
        grads *= img_tensor[0].detach().cpu().numpy()

    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads


def prp(
        ppnet: nn.Module,
        img_tensor: torch.Tensor,
        proto_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
) -> np.array:
    """ Perform patch visualization using SmoothGrad

    :param ppnet: ProtoPnet
    :param img_tensor: Input image tensor
    :param proto_id: Prototype index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :return: gradient map
    """
    if location is None:
        # Find location of feature vector closest to target node
        with torch.no_grad():
            _, distances_batch = ppnet.push_forward(img_tensor)  # Shapes N x P x H x W
        distances_batch = distances_batch[0, proto_id].cpu().numpy()  # Shape H x W
        (h, w) = np.where(distances_batch == np.min(distances_batch))
        h, w = h[0], w[0]
    else:
        # Location is predefined
        h, w = location

    ppnet.eval()
    img_tensor.requires_grad = True

    with torch.enable_grad():
        conv_features = ppnet.conv_features(img_tensor)

        newl2 = l2_lrp_class.apply
        similarities = newl2(conv_features, ppnet)  # Shape N x P x H x W
        min_distances = similarities[:,:,h,w]

    '''For individual prototype'''
    (min_distances[:, proto_id]).backward()
    grads = img_tensor.grad.data[0].detach().cpu().numpy()
    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads

def randgrads(
        ppnet: nn.Module,
        img_tensor: torch.Tensor,
        proto_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
        grad_x_input: Optional[bool] = False,
) -> np.array:
    """ Perform patch visualization using RandGrads

    :param ppnet: ProtoPnet
    :param img_tensor: Input image tensor
    :param proto_id: Prototype index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param grad_x_input: Multiply gradient by input
    :return: gradient map
    """
    grads = np.random.random(img_tensor.shape[1:])
    if grad_x_input:
        grads *= img_tensor[0].detach().cpu().numpy()

    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    # if gaussian_ksize:
    #     grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

from prototree.prototree import ProtoTree
from features.prp import canonize_tree, l2_lrp_class


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
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        node_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = None,
        select_highest: Optional[bool] = True,
) -> np.array:
    """ Perform patch visualization using Cubic interpolation

        :param tree: Prototree
        :param img_tensor: Input image tensor
        :param node_id: Node index
        :param location: Coordinates of feature vector
        :param device: Target device (unused)
        :param select_highest: Keep only location of the highest activation
        :return: interpolated similarity map
    """
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(img_tensor)
        sim_map = torch.exp(-distances_batch[0, node_id]).cpu().numpy()
    if not select_highest:
        return sim_map
    if location is None:
        # Find location of feature vector with the highest similarity
        h, w = np.where(sim_map == np.max(sim_map))
        h, w = h[0], w[0]
    else:
        # Location is predefined
        h, w = location
    masked_similarity_map = np.zeros(sim_map.shape)
    masked_similarity_map[h, w] = 1
    return masked_similarity_map


def smoothgrads(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        node_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
        nsamples: Optional[int] = 10,
        noise: Optional[float] = 0.2,
        grads_x_input: Optional[bool] = True,
) -> np.array:
    """ Perform patch visualization using SmoothGrad

    :param tree: Prototree
    :param img_tensor: Input image tensor
    :param node_id: Node index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param nsamples: Number of samples
    :param noise: Noise level
    :param grads_x_input: Use gradient times input mode
    :return: gradient map
    """
    if location is None:
        # Find location of feature vector closest to target node
        with torch.no_grad():
            _, distances_batch, _ = tree.forward_partial(img_tensor)
        distances_batch = distances_batch[0, node_id].cpu().numpy()  # Shape H x W
        (h, w) = np.where(distances_batch == np.min(distances_batch))
        h, w = h[0], w[0]
    else:
        # Location is predefined
        h, w = location

    if nsamples == 1:
        noisy_images = [img_tensor]
    else:
        # Compute variance from noise ratio
        sigma = (img_tensor.max() - img_tensor.min()).detach().cpu().numpy() * noise
        # Generate noisy images around original.
        noisy_images = [img_tensor + torch.randn(img_tensor.shape).to(device) * sigma for _ in range(nsamples)]

    # Compute gradients
    grads = []
    for x in noisy_images:
        x.requires_grad_()
        # Forward pass
        _, distances_batch, _ = tree.forward_partial(x)
        # Identify target location before backward pass
        output = distances_batch[0, node_id, h, w]
        output.backward(retain_graph=True)
        grads.append(x.grad.data[0].detach().cpu().numpy())

    # grads has shape (nsamples) x img_tensor.shape => average across all samples
    grads = np.mean(np.array(grads), axis=0)
    if grads_x_input:
        grads *= img_tensor[0].detach().cpu().numpy()

    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads

def randgrads(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        node_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
        grads_x_input: Optional[bool] = False,
) -> np.array:
    """ Perform patch visualization using random gradients

    :param tree: Prototree
    :param img_tensor: Input image tensor
    :param node_id: Node index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :param grads_x_input: Use gradient times input mode
    :return: gradient map
    """
    grads = np.random.random(img_tensor.shape[1:])
    if grads_x_input:
        grads *= img_tensor[0].detach().cpu().numpy()

    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads


def prp(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        node_id: int,
        location: Tuple[int, int] = None,
        device: Optional[str] = 'cuda:0',
        polarity: Optional[str] = 'absolute',
        gaussian_ksize: Optional[int] = 5,
        normalize: Optional[bool] = False,
) -> np.array:
    """ Perform patch visualization using Prototyp Relevance Propagation
        (https://www.sciencedirect.com/science/article/pii/S0031320322006513#bib0030)

    :param tree: Prototree
    :param img_tensor: Input image tensor
    :param node_id: Node index
    :param location: Coordinates of feature vector
    :param device: Target device
    :param polarity: Polarity filter applied on gradients
    :param gaussian_ksize: Size of Gaussian filter kernel
    :param normalize: Perform min-max normalization on gradients
    :return: gradient map
    """
    if not hasattr(tree, 'epsilon'):
        tree = canonize_tree(tree, arch='resnet50', device=device)
    tree.eval()
    img_tensor.requires_grad = True

    with torch.enable_grad():
        # Partial forward
        features = tree._net(img_tensor)
        features = tree._add_on(features)
        newl2 = l2_lrp_class.apply
        similarities = newl2(features, tree)
        if location is not None:
            h, w = location
            min_distances = similarities[:, :, h, w]
        else:
            # global max pooling
            min_distances = tree.max_layer(similarities)
            min_distances = min_distances.view(-1, tree.num_prototypes)
        '''For individual prototype'''
        (min_distances[:, node_id]).backward()
    grads = img_tensor.grad.data[0].detach().cpu().numpy()
    # Post-processing
    grads = polarity_and_collapse(grads, polarity=polarity, avg_chan=0)
    if gaussian_ksize:
        grads = gaussian_filter(grads, sigma=gaussian_ksize)
    if normalize:
        grads = normalize_min_max(grads)
    return grads

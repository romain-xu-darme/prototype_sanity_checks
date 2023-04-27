import os
from subprocess import check_call
from PIL import Image
from typing import List
from prototree.upsample import upsample_similarity_map
import torch
from prototree.prototree import ProtoTree


def upsample_local(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        img_path: str,
        seg_path: str,
        output_dir: str,
        decision_path: list,
        threshold: str,
        mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> List[float]:
    """ Given a test sample, compute and store visual representation of parts similar to prototypes

    :param tree: ProtoTree
    :param img_tensor: Input image tensor
    :param img_path: Path to the original image
    :param seg_path: Path to segmentation of the original image (if any)
    :param output_dir: Directory where to store the visualization
    :param decision_path: List of main nodes leading to the prediction
    :param threshold: Upsampling threshold
    :param mode: Either "vanilla" or "smoothgrads"
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    :returns: List of overlap ratios between the prototypes bounding boxes and the object when seg_path is given,
        None otherwise
    """
    os.makedirs(output_dir, exist_ok=True)

    overlaps = []
    for node in decision_path[:-1]:
        overlaps.append(upsample_similarity_map(
            tree=tree,
            img=Image.open(img_path),
            seg=Image.open(seg_path).convert('RGB') if seg_path else None,
            img_tensor=img_tensor,
            node_id=tree._out_map[node],
            node_name=node.index,
            output_dir=output_dir,
            threshold=threshold,
            location=None,  # Upsample location maximizing similarity
            mode=mode,
            grads_x_input=grads_x_input,
        ))
    return overlaps if seg_path else None


def gen_pred_vis(
        tree: ProtoTree,
        img_tensor: torch.Tensor,
        img_path: str,
        seg_path: str,
        proj_dir: str,
        output_dir: str,
        classes: tuple,
        upsample_threshold: str,
        upsample_mode: str = 'vanilla',
        grads_x_input: bool = False,
) -> float:
    """ Generate prediction visualization

    :param tree: ProtoTree
    :param img_tensor: Input image tensor
    :param img_path: Path to the original image
    :param seg_path: Path to the segmentation of the original image (if any)
    :param proj_dir: Directory containing the prototypes projection and visualizations
    :param output_dir: Directory where to store the visualization
    :param classes: Class labels
    :param upsample_threshold: Upsampling threshold
    :param upsample_mode: Either "vanilla" or "smoothgrads"
    :param grads_x_input: Use gradients x image to mask out parts of the image with low gradients
    :returns: Average percentage of overlap between parts positively compared and object segmentation
    """
    assert upsample_mode in ['vanilla', 'smoothgrads', 'prp',], f'Unsupported upsample mode {upsample_mode}'

    # Create directory to store visualization
    img_name = img_path.split('/')[-1].split(".")[-2]
    output_dir = os.path.join(output_dir, img_name)
    os.makedirs(output_dir, exist_ok=True)
    local_upsample_path = os.path.join(output_dir, 'upsampling')

    # Get the model prediction
    with torch.no_grad():
        pred_kwargs = dict()
        pred, pred_info = tree.forward(img_tensor, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()
        assert 'out_leaf_ix' in pred_info.keys()

    # Copy input image
    sample_path = os.path.join(output_dir, 'sample.jpg')
    Image.open(img_path).save(sample_path)

    leaf_ix = pred_info['out_leaf_ix'][0]
    leaf = tree.nodes_by_index[leaf_ix]
    decision_path = tree.path_to(leaf)

    overlaps = upsample_local(
        tree=tree,
        img_tensor=img_tensor,
        img_path=img_path,
        seg_path=seg_path,
        output_dir=local_upsample_path,
        decision_path=decision_path,
        threshold=upsample_threshold,
        mode=upsample_mode,
        grads_x_input=grads_x_input
    )

    avg_overlap = 1.0
    if overlaps is not None:
        # Compute stats on percentage of overlap between part visualizations and object segmentation
        npos = 0  # Number of positive comparisons
        sum_overlap = 0.0  # Cumulative percentage of overlap
        for i, node in enumerate(decision_path[:-1]):
            node_ix = node.index
            prob = probs[node_ix].item()
            if prob > 0.5:
                sum_overlap += overlaps[i]
                npos += 1
        avg_overlap = sum_overlap/npos if npos else 1.0

    # Prediction graph is visualized using Graphviz
    # Build dot string
    s = 'digraph T {margin=0;rankdir=LR\n'
    s += 'node [shape=plaintext, label=""];\n'
    s += 'edge [penwidth="0.5"];\n'

    # Create a node for the sample image
    s += f'sample[image="{sample_path}"];\n'

    # Create nodes for all decisions/branches
    # Starting from the leaf
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()

        s += f'node_{i + 1}[image="{proj_dir}/upsampling/{node_ix}_nearest_patch_of_image.png" ' \
             f'group="{"g" + str(i)}"];\n'
        if prob > 0.5:
            s += f'node_{i + 1}_original[image="{local_upsample_path}/' \
                 f'{node_ix}_bounding_box_nearest_patch_of_image.png" imagescale=width group="{"g" + str(i)}"];\n'
            label = "Present      \nSimilarity %.4f                   " % prob
            if overlaps is not None:
                label += "\nOverlap %.2f               " % overlaps[i]
            s += f'node_{i + 1}->node_{i + 1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        else:
            s += f'node_{i + 1}_original[image="{sample_path}" group="{"g" + str(i)}"];\n'
            label = "Absent      \nSimilarity %.4f                   " % prob
            s += f'node_{i + 1}->node_{i + 1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'

        s += f'node_{i + 1}->node_{i + 2};\n'
        s += "{rank = same; "f'node_{i + 1}_original' + "; " + f'node_{i + 1}' + "};"

    # Create a node for the model output
    s += f'node_{len(decision_path)}[imagepos="tc" imagescale=height image="{proj_dir}/node_vis/' \
         f'node_{leaf_ix}_vis.jpg" label="{classes[label_ix]}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'

    # Connect the input image to the first decision node
    s += 'sample->node_1;\n'

    s += '}\n'

    with open(os.path.join(output_dir, 'predvis.dot'), 'w') as f:
        f.write(s)

    from_p = os.path.join(output_dir, 'predvis.dot')
    to_pdf = os.path.join(output_dir, 'predvis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s' % (from_p, to_pdf), shell=True)
    return avg_overlap

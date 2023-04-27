from util.args import *
from util.data import get_dataloaders
from util.visualize import gen_vis
from util.save import *
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from prototree.upsample import upsample_prototypes
from util.analyse import average_distance_nearest_image

import torch

# Use only deterministic algorithms
torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Finalize a ProtoTree (pruning and projection)')
    add_general_args(parser)
    add_finalize_args(parser)
    return parser


def finalize_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args

    log = Log(args.root_dir, mode='a')
    print("Log dir: ", args.root_dir)
    log.log_message('Device used: ' + args.device)

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(
        dataset=args.dataset,
        projection_mode=args.projection_mode,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Either latest checkpoint or the one pointed by args
    directory_path = log.checkpoint_dir+'/latest' if not args.tree_dir else args.tree_dir
    tree, (optimizer, params_to_freeze, params_to_train), scheduler, stats = \
        load_checkpoint(directory_path)
    best_train_acc, best_test_acc, leaf_labels, epoch = stats
    tree.to(args.device)

    # Pruning
    prune(tree, args.pruning_threshold_leaves, log)
    # Projection
    proj_dir = os.path.join(args.root_dir, args.proj_dir)
    os.makedirs(proj_dir, exist_ok=True)
    project_info, tree = project_with_class_constraints(tree, projectloader, args.device, log)
    average_distance_nearest_image(project_info, tree, log)
    save_checkpoint(f'{proj_dir}/model/',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)

    # Upsample prototype for visualization
    upsample_prototypes(
        tree=tree,
        project_info=project_info,
        project_loader=projectloader,
        output_dir=os.path.join(proj_dir, "upsampling"),
        threshold=args.upsample_threshold,
        log=log,
        mode=args.upsample_mode,
        grads_x_input=args.grads_x_input,
    )
    # Save projection file
    torch.save(project_info, os.path.join(proj_dir, 'projection.pth'))
    # visualize tree
    gen_vis(tree, classes, proj_dir)


if __name__ == '__main__':
    finalize_tree()

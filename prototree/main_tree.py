from util.args import *
from util.data import get_dataloaders
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval_accuracy, eval_fidelity
from prototree.prune import prune
from prototree.project import project_with_class_constraints
from prototree.upsample import upsample_prototypes

import torch
from copy import deepcopy

# Use onyl deterministic algorithms
torch.use_deterministic_algorithms(True)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Train a ProtoTree')
    add_prototree_init_args(parser)
    add_general_args(parser)
    add_training_args(parser)
    return parser


def run_tree(args: argparse.Namespace = None):
    args = get_args(create_parser()) if args is None else args

    resume = False
    if (os.path.exists(args.root_dir) and os.path.exists(args.root_dir+'/metadata')
            and load_args(args.root_dir+'/metadata') == args and os.path.exists(args.root_dir+'/checkpoints/latest')) \
            or args.tree_dir != '':
        # Directory already exists and contains the same arguments => resume computation
        # Alternatively, checkpoint can be explicitely specified
        resume = True

    if os.path.exists(args.root_dir) and not resume and not args.force:
        raise ValueError(f'Output directory {args.root_dir} already exists. To overwrite, use --force option.')

    # Create a logger
    log = Log(args.root_dir, mode='a' if resume else 'w')
    print("Log dir: ", args.root_dir, flush=True)
    # Log the run arguments
    save_args(args, log.metadata_dir)
    device = args.device

    # Log which device was actually used
    log.log_message('Device used: ' + device)

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(
        dataset=args.dataset,
        projection_mode=args.projection_mode,
        batch_size=args.batch_size,
        device=args.device,
    )

    if not resume:
        # Create a convolutional network based on arguments and add 1x1 conv layer
        features_net, add_on_layers = get_network(
            net=args.net,
            init_mode=args.init_mode,
            num_features=args.num_features,
        )

        # Create a ProtoTree
        tree = ProtoTree(
            num_classes=len(classes),
            depth=args.depth,
            num_features=args.num_features,
            features_net=features_net,
            add_on_layers=add_on_layers,
            derivative_free=not args.disable_derivative_free_leaf_optim,
            kontschieder_normalization=args.kontschieder_normalization,
            kontschieder_train=args.kontschieder_train,
            log_probabilities=args.log_probabilities,
            focal_distance=args.focal_distance,
            H1=args.H1,
            W1=args.W1,
        )
        tree = tree.to(device)
        # Determine which optimizer should be used to update the tree parameters
        optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones,
                                                         gamma=args.gamma)
        log.log_message(
            "Max depth %s, so %s internal nodes and %s leaves" % (args.depth, tree.num_branches, tree.num_leaves))
        analyse_output_shape(tree, trainloader, log, device)

        leaf_labels = dict()
        best_train_acc = 0.
        best_test_acc = 0.

        save_checkpoint(
            f'{log.checkpoint_dir}/tree_init', tree, optimizer, scheduler, 0,
            best_train_acc, best_test_acc, leaf_labels, args)
        epoch = 1
    else:
        # Either latest checkpoint or the one pointed by args
        directory_path = log.checkpoint_dir+'/latest' if not args.tree_dir else args.tree_dir
        print('Resuming computation from ' + directory_path)
        tree, (optimizer, params_to_freeze, params_to_train), scheduler, stats = \
            load_checkpoint(directory_path)
        tree.to(device)
        best_train_acc, best_test_acc, leaf_labels, epoch = stats
        # Go to the next epoch
        epoch += 1

    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    logged_values = ('test_acc', 'mean_total_loss', 'mean_train_acc')
    log.create_log('log_epoch_overview', 'epoch', *logged_values)

    if epoch < args.epochs+1:
        '''
            TRAIN AND EVALUATE TREE
        '''
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s" % str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(epoch, params_to_freeze, params_to_train, args.freeze_epochs, log)
            log_learning_rates(optimizer, args, log)

            # Train tree
            if tree._kontschieder_train:
                train_info = train_epoch_kontschieder(
                    tree, trainloader, optimizer, epoch,
                    args.disable_derivative_free_leaf_optim, device)
            else:
                train_info = train_epoch(
                    tree, trainloader, optimizer, epoch,
                    args.disable_derivative_free_leaf_optim, device)
            # Update scheduler and leaf labels before saving checkpoints
            scheduler.step()
            leaf_labels = analyse_leafs(tree, epoch, len(classes), leaf_labels, args.pruning_threshold_leaves, log)

            # Update best train accuracy (if necessary)
            best_train_acc = save_best_train_tree(
                tree, optimizer, scheduler, epoch,
                train_info['train_accuracy'], best_train_acc, best_test_acc, leaf_labels, args, log)
            save_tree(
                tree, optimizer, scheduler, epoch,
                best_train_acc, best_test_acc, leaf_labels, args, log)

            # Evaluate tree
            if args.epochs <= 150 or epoch % 10 == 0 or epoch == args.epochs:
                eval_info = eval_accuracy(tree, testloader, f'Epoch {epoch}: ', device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_tree(
                    tree, optimizer, scheduler, epoch,
                    best_train_acc, original_test_acc, best_test_acc, leaf_labels, args, log)
                stats = (original_test_acc, train_info['loss'], train_info['train_accuracy'])
                log.log_values('log_epoch_overview', epoch, *stats)
            else:
                stats = ("n.a.", train_info['loss'], train_info['train_accuracy'])
                log.log_values('log_epoch_overview', epoch, *stats)

    else:  # tree was loaded and not trained, so evaluate only
        '''
            EVALUATE TREE
        '''
        # Readjust epoch index
        epoch = args.epochs
        original_test_acc = None
        if not args.skip_eval_after_training:
            eval_info = eval_accuracy(tree, testloader, f'Epoch {epoch}: ', device, log)
            original_test_acc = eval_info['test_accuracy']
            best_test_acc = save_best_test_tree(
                tree, optimizer, scheduler, epoch,
                best_train_acc, original_test_acc, best_test_acc, leaf_labels, args, log)
            stats = (original_test_acc, "n.a.", "n.a.")
            log.log_values('log_epoch_overview', epoch, *stats)

    '''
        EVALUATE AND ANALYSE TRAINED TREE
    '''
    log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n"
                    % (str(best_train_acc), str(best_test_acc)))
    trained_tree = deepcopy(tree)
    leaf_labels = analyse_leafs(tree, epoch+1, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)

    '''
        PRUNE
    '''
    prune(tree, args.pruning_threshold_leaves, log)
    save_checkpoint(f'{log.checkpoint_dir}/pruned',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    pruned_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree
    leaf_labels = analyse_leafs(tree, epoch+2, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    pruned_test_acc = None
    if not args.skip_eval_after_training:
        eval_info = eval_accuracy(tree, testloader, "Pruned tree", device, log)
        pruned_test_acc = eval_info['test_accuracy']

    '''
        PROJECT
    '''
    proj_dir = os.path.join(args.root_dir, args.proj_dir)
    os.makedirs(proj_dir, exist_ok=True)
    project_info, tree = project_with_class_constraints(tree, projectloader, device, log)
    save_checkpoint(f'{proj_dir}/model/',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    analyse_leafs(tree, epoch+3, len(classes), leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(tree, log)
    pruned_projected_test_acc = eval_info_samplemax = eval_info_greedy = fidelity_info = None
    if not args.skip_eval_after_training:
        eval_info = eval_accuracy(tree, testloader, "Pruned and projected", device, log)
        pruned_projected_test_acc = eval_info['test_accuracy']
        eval_info_samplemax = eval_accuracy(tree, testloader, "Pruned and projected", device, log, 'sample_max')
        get_avg_path_length(tree, eval_info_samplemax, log)
        eval_info_greedy = eval_accuracy(tree, testloader, "Pruned and projected", device, log, 'greedy')
        get_avg_path_length(tree, eval_info_greedy, log)
        fidelity_info = eval_fidelity(tree, testloader, device, log)

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

    return trained_tree.to('cpu'), pruned_tree.to('cpu'), pruned_projected_tree.to('cpu'), \
        original_test_acc, pruned_test_acc, pruned_projected_test_acc, \
        project_info, eval_info_samplemax, eval_info_greedy, fidelity_info


if __name__ == '__main__':
    run_tree()

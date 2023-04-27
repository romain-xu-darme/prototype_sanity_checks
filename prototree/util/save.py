import torch
import random
import numpy as np
import os
import argparse
import pickle
from prototree.prototree import ProtoTree
from util.log import Log
from util.args import save_args, load_args, get_optimizer
from typing import Tuple, Any, List


def save_checkpoint(
        directory_path: str,
        tree: ProtoTree,
        optimizer,
        scheduler,
        epoch: int,
        best_train_acc: float,
        best_test_acc: float,
        leaf_labels: dict,
        args: argparse.Namespace):
    """ Save everything needed to restart a training process

    :param directory_path: Target location
    :param tree: ProtoTree
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param epoch: Current epoch
    :param best_train_acc: Current best training accuracy
    :param best_test_acc: Current best test accuracy
    :param leaf_labels: Current leaf labels
    :param args: Command line arguments
    """
    os.makedirs(directory_path, exist_ok=True)
    tree.eval()
    tree.save(directory_path)
    tree.save_state(directory_path)
    torch.save(optimizer.state_dict(), f'{directory_path}/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{directory_path}/scheduler_state.pth')
    # Save random state
    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    stats = {'best_train_acc': best_train_acc,
             'best_test_acc': best_test_acc,
             'leaf_labels': leaf_labels,
             'epoch': epoch,
             'torch_random_state': torch_rng_state,
             'numpy_random_state': numpy_rng_state,
             'python_random_state': python_rng_state}
    with open(f'{directory_path}/stats.pickle', 'wb') as f:
        pickle.dump(stats, f)
    save_args(args, directory_path)


def load_checkpoint(directory_path: str) -> \
        Tuple[ProtoTree, Tuple[Any, List, List], Any, Tuple[float, float, dict, int]]:
    """ Restore training process using checkpoint directory.

    :param directory_path: Target location
    :returns: tree, (optimizer, params_to_freeze, params_to_train), scheduler,
        (best_train_acc, best_test_acc, leaf_labels, epoch)
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Unknown checkpoint directory {directory_path}")

    # Load all arguments
    args = load_args(directory_path)
    if not hasattr(args, 'device'):
        args.device = 'cpu'

    # Load tree, optimizer, scheduler
    tree = ProtoTree.load(directory_path, map_location=args.device)
    optimizer, params_to_freeze, params_to_train = get_optimizer(tree, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    optimizer.load_state_dict(torch.load(f'{directory_path}/optimizer_state.pth', map_location=args.device))
    scheduler.load_state_dict(torch.load(f'{directory_path}/scheduler_state.pth', map_location=args.device))

    # Recover current training stats
    with open(directory_path + '/stats.pickle', 'rb') as f:
        stats = pickle.load(f)
    best_train_acc, best_test_acc, leaf_labels, epoch, torch_rng, numpy_rng, python_rng = tuple(stats.values())
    # Resume random states
    torch.random.set_rng_state(torch_rng)
    np.random.set_state(numpy_rng)
    random.setstate(python_rng)
    print(f"Checkpoint loaded. Epoch: {epoch}, best train acc: {best_train_acc}, best test acc: {best_test_acc}")
    return tree, (optimizer, params_to_freeze, params_to_train), scheduler, \
        (best_train_acc, best_test_acc, leaf_labels, epoch)


def save_tree(
        tree: ProtoTree,
        optimizer,
        scheduler,
        epoch: int,
        best_train_acc: float,
        best_test_acc: float,
        leaf_labels: dict,
        args: argparse.Namespace,
        log: Log,
        checkpoint_frequency: int = 10,
):
    assert checkpoint_frequency > 0, f'Invalid checkpoint frequency {checkpoint_frequency}'
    # Save latest model
    save_checkpoint(f'{log.checkpoint_dir}/latest',
                    tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)

    # Save model every 10 epochs
    if epoch % checkpoint_frequency == 0:
        save_checkpoint(f'{log.checkpoint_dir}/epoch_{epoch}',
                        tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)


def save_best_train_tree(
        tree: ProtoTree,
        optimizer,
        scheduler,
        epoch: int,
        train_acc: float,
        best_train_acc: float,
        best_test_acc: float,
        leaf_labels: dict,
        args: argparse.Namespace,
        log: Log,
):
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        save_checkpoint(f'{log.checkpoint_dir}/best_train_model',
                        tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    return best_train_acc


def save_best_test_tree(
        tree: ProtoTree,
        optimizer,
        scheduler,
        epoch: int,
        best_train_acc: float,
        test_acc: float,
        best_test_acc: float,
        leaf_labels: dict,
        args: argparse.Namespace,
        log: Log,
):
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_checkpoint(f'{log.checkpoint_dir}/best_test_model',
                        tree, optimizer, scheduler, epoch, best_train_acc, best_test_acc, leaf_labels, args)
    return best_test_acc

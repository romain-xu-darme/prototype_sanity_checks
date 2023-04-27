from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from prototree.prototree import ProtoTree


def train_epoch(
        tree: ProtoTree,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        disable_derivative_free_leaf_optim: bool,
        device: str,
        progress_prefix: str = 'Train Epoch',
) -> dict:

    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        ys_pred, info = tree.forward(xs)

        # Learn prototypes and network with gradient descent.
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)

        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        if not disable_derivative_free_leaf_optim:
            # Update leaves with derivate-free algorithm
            # Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys]  # shape (batchsize, num_classes)
                for leaf in tree.leaves:
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(
                            torch.logsumexp(info['pa_tensor'][leaf.index]
                                            + leaf.distribution()
                                            + torch.log(target)
                                            - ys_pred,
                                            dim=0))
                    else:
                        update = torch.sum(
                            (info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred,
                            dim=0)
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    # dist_params values can get slightly negative because of floating point issues.
                    # therefore, set to zero.
                    F.relu_(leaf._dist_params)
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        postfix_str = f'Batch [{i + 1}/{len(train_loader)}], ' \
                      f'CCE loss: {loss.item():.3f}, Acc: {acc:.3f}'
        train_iter.set_postfix_str(postfix_str)
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

    train_info['loss'] = total_loss/nr_batches
    train_info['train_accuracy'] = total_acc/nr_batches
    return train_info


def train_epoch_kontschieder(
        tree: ProtoTree,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        disable_derivative_free_leaf_optim: bool,
        device: str,
        progress_prefix: str = 'Train Epoch',
) -> dict:

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    nr_batches = float(len(train_loader))

    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
    else:
        if tree._kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network.
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        ys_pred, _ = tree.forward(xs)

        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)

        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        postfix_str = f'Batch [{i + 1}/{len(train_loader)}], ' \
                      f'CCE loss: {loss.item():.3f}, Acc: {acc:.3f}'
        train_iter.set_postfix_str(postfix_str)
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

    train_info['loss'] = total_loss/nr_batches
    train_info['train_accuracy'] = total_acc/nr_batches
    return train_info


# Updates leaves with derivative-free algorithm
def train_leaves_epoch(
        tree: ProtoTree,
        train_loader: DataLoader,
        epoch: int,
        device: str,
        progress_prefix: str = 'Train Leafs Epoch',
) -> None:
    # Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

        # Show progress on progress bar
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=progress_prefix+' %s' % epoch, ncols=0)

        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf._dist_params)

        for i, (xs, ys) in train_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Train leafs without gradient descent
            out, info = tree.forward(xs)
            target = eye[ys]  # shape (batchsize, num_classes)
            for leaf in tree.leaves:
                if tree._log_probabilities:
                    # log version
                    update = torch.exp(
                        torch.logsumexp(
                            info['pa_tensor'][leaf.index]
                            + leaf.distribution()
                            + torch.log(target)
                            - out,
                            dim=0))
                else:
                    update = torch.sum(
                        (info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out,
                        dim=0)
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf._dist_params -= leaf._dist_params  # set current dist params to zero
            leaf._dist_params += update_sum[leaf]  # give dist params new value

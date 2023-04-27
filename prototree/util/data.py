import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


def get_data(
        dataset: str,
        training_mode: str,
        projection_mode: str = None,
        dir_path: str = '../data',
) -> Tuple:
    """
    Load the proper dataset based on the parsed arguments

    :param dataset: Name of the dataset
    :param training_mode: Either "corners", "cropped" or "full"
    :param projection_mode: Either "corners", "cropped" or "raw"
    :param dir_path: Dataset root directory
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """

    if dataset == 'CUB-200-2011':
        mode_path = {
            "corners": dir_path+'/CUB_200_2011/dataset/train_corners',
            "cropped": dir_path+'/CUB_200_2011/dataset/train_crop',
            "raw": dir_path+'/CUB_200_2011/dataset/train_full',
            None: None,
        }
        project_path = mode_path[projection_mode]
        train_path = mode_path[training_mode]
        return get_birds(augment=True,
                         train_dir=train_path,
                         project_dir=project_path,
                         test_dir=dir_path+'/CUB_200_2011/dataset/test_full')
    # FIXME : Remove
    if dataset == 'CUB-small':
        return get_birds(augment=True,
                         train_dir=dir_path+'/CUB_200_2011/dataset/train_small',
                         project_dir=dir_path+'/CUB_200_2011/dataset/train_small',
                         test_dir=dir_path+'/CUB_200_2011/dataset/train_small')
    if dataset == 'CARS':
        return get_cars(
            augment=True,
            train_dir=dir_path+'/cars/dataset/train',
            project_dir=dir_path+'/cars/dataset/train',
            test_dir=dir_path+'/cars/dataset/test')
    raise Exception(f'Could not load data set "{dataset}"!')


def get_dataloaders(dataset: str, projection_mode: str, batch_size: int, device: str):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape = get_data(
        dataset=dataset,
        training_mode="corners",
        projection_mode=projection_mode,
        dir_path='../data'
    )
    c, w, h = shape
    # Determine if GPU should be used
    cuda = device.startswith('cuda')
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    # make batch size smaller to prevent out of memory errors during projection
    projectloader = torch.utils.data.DataLoader(projectset,
                                                batch_size=int(batch_size / 4),
                                                shuffle=False,
                                                pin_memory=cuda
                                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c


def get_birds(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, shear=(-2, 2), translate=[0.05, 0.05]),
            ]),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment) if project_dir else None
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    for i in range(len(classes)):
        classes[i] = classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


def get_cars(augment: bool, train_dir: str, project_dir: str, test_dir: str, img_size=224):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),  # resize to 256x256
            transforms.RandomOrder([
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.4, 0.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=15, shear=(-2, 2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)),  # crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes

    return trainset, projectset, testset, classes, shape

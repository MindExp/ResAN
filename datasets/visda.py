import os

import torch
from torchvision import transforms

from datasets.utils import folder

root_file_path = 'E:\Projects\dataset\VisDA2017'


def get_loader_mcd(config):
    dataset_train, dataset_test = load_visda_mcd(config)

    dataloader_trainset = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=config.batch_size, shuffle=True, drop_last=True,
    )
    dataloader_testset = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=config.batch_size, shuffle=False, drop_last=False,
    )

    return dataloader_trainset, dataloader_testset


def load_visda_mcd(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dataset_train = folder.ImageFolder(root=os.path.join(root_file_path, 'train'), transform=transform_train)
    dataset_test = folder.ImageFolder(root=os.path.join(root_file_path, 'validation'), transform=transform_test)
    print('Dataset Size: source domain: {}||target domain: {}'.format(len(dataset_train), len(dataset_test)))
    print('Classes: {}'.format(str(dataset_train.classes)))

    return dataset_train, dataset_test

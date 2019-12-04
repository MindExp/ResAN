import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.io import loadmat

from datasets.utils.unaligned_data_loader import UnalignedDataLoader
from utils import util

file_path_datasets_root = 'E:\Projects\dataset'
file_path_svhn_train_mcd = os.path.join(file_path_datasets_root, 'svhn\\train_32x32.mat')
file_path_svhn_test_mcd = os.path.join(file_path_datasets_root, 'svhn\\test_32x32.mat')
file_path_svhn_train_gta = os.path.join(file_path_datasets_root, 'svhn\\trainset')
file_path_svhn_test_gta = os.path.join(file_path_datasets_root, 'svhn\\testset')


def get_loader_gta(config):
    dataloader_trainset, dataloader_testset = load_svhn_gta(config)
    return dataloader_trainset, dataloader_testset


def get_loader_mcd(config):
    train_set = {}
    test_set = {}

    train_set_data, train_set_label, test_set_data, test_set_label = load_svhn_mcd()

    train_set['imgs'] = train_set_data
    train_set['labels'] = train_set_label
    test_set['imgs'] = test_set_data
    test_set['labels'] = test_set_label

    scale = 40 if config.source == 'synth' else 28 if config.source == 'usps' or config.target == 'usps' else 32

    data_loader = UnalignedDataLoader()
    data_loader.initialize(train_set, test_set, config.batch_size, scale=scale)
    dataloader_trainset, dataloader_testset = data_loader.load_data()
    return dataloader_trainset, dataloader_testset


# TODO 自定义返回 id 信息 ImageFolder, def __getitem__(self, index):
def load_svhn_gta(config):
    file_path_trainset = file_path_svhn_train_gta
    file_path_testset = file_path_svhn_test_gta
    mean = np.array([0.44, 0.44, 0.44])
    std = np.array([0.19, 0.19, 0.19])

    transform_trainset = transforms.Compose([transforms.Resize(config.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    transform_testset = transform_trainset

    dataset_trainset = datasets.ImageFolder(root=file_path_trainset, transform=transform_trainset)
    dataset_testset = datasets.ImageFolder(root=file_path_testset, transform=transform_testset)

    dataloader_trainset = torch.utils.data.DataLoader(dataset_trainset,
                                                      batch_size=config.batch_size,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=True)
    dataloader_testset = torch.utils.data.DataLoader(dataset_testset,
                                                     batch_size=config.batch_size,
                                                     shuffle=False,
                                                     num_workers=2,
                                                     drop_last=False)
    return dataloader_trainset, dataloader_testset


def load_svhn_mcd():
    train_set = loadmat(file_path_svhn_train_mcd)
    test_set = loadmat(file_path_svhn_test_mcd)
    train_set_data = train_set['X']
    train_set_data = train_set_data.transpose(3, 2, 0, 1).astype(np.float32)
    train_set_label = util.dense_to_one_hot(train_set['y'])
    test_set_data = test_set['X']
    test_set_data = test_set_data.transpose(3, 2, 0, 1).astype(np.float32)
    test_set_label = util.dense_to_one_hot(test_set['y'])

    return train_set_data, train_set_label, test_set_data, test_set_label

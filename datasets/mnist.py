import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.io import loadmat
from datasets.utils.unaligned_data_loader import UnalignedDataLoader
import os

file_path_datasets_root = 'D:\Exp_HZH\dataset'
file_path_mnist_mcd = os.path.join(file_path_datasets_root, 'mnist\mnist_data.mat')
file_path_mnist_train_gta = os.path.join(file_path_datasets_root, 'E:\Projects\dataset\mnist\\trainset')
file_path_mnist_test_gta = os.path.join(file_path_datasets_root, 'E:\Projects\dataset\mnist\\testset')


def get_loader_gta(config):
    dataloader_trainset, dataloader_testset = load_mnist_gta(config)
    return dataloader_trainset, dataloader_testset


def get_loader_mcd(config):
    train_set = {}
    test_set = {}

    train_data, train_label, test_data, test_label = load_mnist_mcd(
        usps=True if config.source == 'usps' or config.target == 'usps' else False, all_use=config.all_use)

    train_set['imgs'] = train_data
    train_set['labels'] = train_label
    test_set['imgs'] = test_data
    test_set['labels'] = test_label

    scale = 40 if config.source == 'synth' else 28 if config.source == 'usps' or config.target == 'usps' else 32

    data_loader = UnalignedDataLoader()
    data_loader.initialize(train_set, test_set, config.batch_size, scale=scale)
    dataloader_trainset, dataloader_testset = data_loader.load_data()

    return dataloader_trainset, dataloader_testset


def load_mnist_gta(config):
    file_path_trainset = file_path_mnist_train_gta
    file_path_testset = file_path_mnist_test_gta
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


def load_mnist_mcd(scale=True, usps=False, all_use=False):
    mnist_data = loadmat(file_path_mnist_mcd)
    if scale:
        train_set_data = np.reshape(mnist_data['train_32'], (55000, 32, 32, 1))
        test_set_data = np.reshape(mnist_data['test_32'], (10000, 32, 32, 1))
        train_set_data = np.concatenate([train_set_data, train_set_data, train_set_data], 3)
        test_set_data = np.concatenate([test_set_data, test_set_data, test_set_data], 3)
        train_set_data = train_set_data.transpose(0, 3, 1, 2).astype(np.float32)
        test_set_data = test_set_data.transpose(0, 3, 1, 2).astype(np.float32)
        train_set_label = mnist_data['label_train']
        test_set_label = mnist_data['label_test']
    else:
        train_set_data = mnist_data['train_28']
        test_set_data = mnist_data['test_28']
        train_set_label = mnist_data['label_train']
        test_set_label = mnist_data['label_test']
        train_set_data = train_set_data.astype(np.float32)
        test_set_data = test_set_data.astype(np.float32)
        train_set_data = train_set_data.transpose((0, 3, 1, 2))
        test_set_data = test_set_data.transpose((0, 3, 1, 2))
    train_set_label = np.argmax(train_set_label, axis=1)
    inds = np.random.permutation(train_set_data.shape[0])
    train_set_data = train_set_data[inds]
    train_set_label = train_set_label[inds]
    test_set_label = np.argmax(test_set_label, axis=1)
    if usps and not all_use:
        train_set_data = train_set_data[:2000]
        train_set_label = train_set_label[:2000]

    return train_set_data, train_set_label, test_set_data, test_set_label

import gzip
import os
import pickle as cPickle

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from datasets.utils.unaligned_data_loader import UnalignedDataLoader

file_path_datasets_root = 'D:\Exp_HZH\dataset'
file_path_usps_mcd = os.path.join(file_path_datasets_root, 'usps\\usps_28x28.pkl')
root_path_usps_cogan = os.path.join(file_path_datasets_root, 'usps\\usps_cogan')


def get_loader_mcd(config):
    train_set = {}
    test_set = {}

    train_data, train_label, test_data, test_label = load_usps_mcd(config.all_use)

    train_set['imgs'] = train_data
    train_set['labels'] = train_label
    test_set['imgs'] = test_data
    test_set['labels'] = test_label

    scale = 40 if config.source == 'synth' else 28 if config.source == 'usps' or config.target == 'usps' else 32

    data_loader = UnalignedDataLoader()
    data_loader.initialize(train_set, test_set, config.batch_size, scale=scale)
    dataloader_trainset, dataloader_testset = data_loader.load_data()

    return dataloader_trainset, dataloader_testset


def load_usps_mcd(use_all=False):
    f = gzip.open(file_path_usps_mcd, 'rb')
    data_set = cPickle.load(f, encoding='bytes')
    f.close()
    train_data = data_set[0][0]
    train_label = data_set[0][1]
    test_data = data_set[1][0]
    test_label = data_set[1][1]
    index = np.random.permutation(train_data.shape[0])
    if use_all:
        train_data = train_data[index][:6562]
        train_label = train_label[index][:6562]
    else:
        train_data = train_data[index][:1800]
        train_label = train_label[index][:1800]
    train_data = train_data * 255
    test_data = test_data * 255
    train_data = train_data.reshape((train_data.shape[0], 1, 28, 28))
    test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))
    return train_data, train_label, test_data, test_label


def get_loader_cogan(config):
    dataset_train, dataset_test = load_usps_cogan(config)
    dataloader_trainset = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    dataloader_testset = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False)

    return dataloader_trainset, dataloader_testset


class USPS_COGAN(torch.utils.data.Dataset):
    def __init__(self, root, dataset_type, transform=None):
        self.transform = transform
        file_path_data_label = None
        if dataset_type == 'trainset':
            self.directory_image = os.path.join(root, 'original_images')
            file_path_data_label = os.path.join(root, "usps_train_list.txt")
        elif dataset_type == 'testset':
            self.directory_image = os.path.join(root, 'original_images')
            file_path_data_label = os.path.join(root, "usps_test_list.txt")

        with open(file_path_data_label, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])),
                                [content_line.strip().split() for content_line in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        image_name, label = self.mapping[index]
        image_path = os.path.join(self.directory_image, image_name)
        image = self.transform(Image.open(image_path).convert('RGB'))
        return index, image, label


def load_usps_cogan(config):

    scale = 40 if config.source == 'synth' else 28 if config.source == 'usps' or config.target == 'usps' else 32
    transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = USPS_COGAN(root=root_path_usps_cogan, dataset_type='trainset', transform=transform)
    dataset_test = USPS_COGAN(root=root_path_usps_cogan, dataset_type='testset', transform=transform)

    return dataset_train, dataset_test

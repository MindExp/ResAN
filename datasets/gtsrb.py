import os
import pickle as pkl

import numpy as np

from datasets.utils.unaligned_data_loader import UnalignedDataLoader

file_path_datasets_root = 'D:\Exp_HZH\dataset'


def get_loader_mcd(config):
    train_set = {}
    test_set = {}

    train_set_data, train_set_label, test_set_data, test_set_label = load_gtsrb()

    train_set['imgs'] = train_set_data
    train_set['labels'] = train_set_label
    test_set['imgs'] = test_set_data
    test_set['labels'] = test_set_label

    scale = 40 if config.source == 'synsig' else 28 if config.source == 'usps' or config.target == 'usps' else 32

    data_loader = UnalignedDataLoader()
    data_loader.initialize(train_set, test_set, config.batch_size, scale=scale)
    dataloader_trainset, dataloader_testset = data_loader.load_data()

    return dataloader_trainset, dataloader_testset


def load_gtsrb():
    file_path_gtsrb = os.path.join(file_path_datasets_root, 'gtsrb\data_gtsrb')
    gtsrb_datasets = pkl.load(open(file_path_gtsrb, 'rb'), encoding='latin1')
    target_train = np.random.permutation(len(gtsrb_datasets['image']))
    train_set_data = gtsrb_datasets['image'][target_train[:31367], :, :, :]
    test_set_data = gtsrb_datasets['image'][target_train[31367:], :, :, :]
    train_set_label = gtsrb_datasets['label'][target_train[:31367]] + 1
    test_set_label = gtsrb_datasets['label'][target_train[31367:]] + 1
    train_set_data = train_set_data.transpose(0, 3, 1, 2).astype(np.float32)
    test_set_data = test_set_data.transpose(0, 3, 1, 2).astype(np.float32)
    return train_set_data, train_set_label, test_set_data, test_set_label

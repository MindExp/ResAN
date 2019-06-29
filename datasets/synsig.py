import os
import pickle as pkl

import numpy as np

from datasets.utils.unaligned_data_loader import UnalignedDataLoader

file_path_datasets_root = 'D:\Exp_HZH\dataset'


def get_loader_mcd(config):
    train_set = {}
    test_set = {}

    train_set_data, train_set_label, test_set_data, test_set_label = load_syntraffic()

    train_set['imgs'] = train_set_data
    train_set['labels'] = train_set_label
    test_set['imgs'] = test_set_data
    test_set['labels'] = test_set_label

    scale = 40 if config.source == 'synsig' else 28 if config.source == 'usps' or config.target == 'usps' else 32

    data_loader = UnalignedDataLoader()
    data_loader.initialize(train_set, test_set, config.batch_size, scale=scale)
    dataloader_trainset, dataloader_testset = data_loader.load_data()

    return dataloader_trainset, dataloader_testset


def load_syntraffic():
    file_path_syntraffic = os.path.join(file_path_datasets_root, 'synthetic_traffic_signs\data_synthetic')
    syn_traffic_datasets = pkl.load(open(file_path_syntraffic, 'rb'), encoding='latin1')
    source_train = np.random.permutation(len(syn_traffic_datasets['image']))
    train_set_data = syn_traffic_datasets['image'][source_train[:len(syn_traffic_datasets['image'])], :, :, :]
    test_set_data = syn_traffic_datasets['image'][source_train[len(syn_traffic_datasets['image']) - 2000:], :, :, :]
    train_set_label = syn_traffic_datasets['label'][source_train[:len(syn_traffic_datasets['image'])]]
    test_set_label = syn_traffic_datasets['label'][source_train[len(syn_traffic_datasets['image']) - 2000:]]
    train_set_data = train_set_data.transpose(0, 3, 1, 2).astype(np.float32)
    test_set_data = test_set_data.transpose(0, 3, 1, 2).astype(np.float32)
    return train_set_data, train_set_label, test_set_data, test_set_label

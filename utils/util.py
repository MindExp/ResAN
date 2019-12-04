import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from datasets import mnist, usps, office_31


def init_weights(m):
    """

    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def dense_to_one_hot(labels_dense):
    """
    Convert class labels from scalars to one-hot vectors.
    :param labels_dense:
    :return:
    """
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def init_record_directory(config):
    svhn_mnist, mnist_usps, synsig_gtsrb, office_31, officeHome = \
        ['svhn', 'mnist'], ['mnist', 'usps'], ['synsig', 'gtsrb'], ['A', 'W', 'D'], ['A', 'C', 'P', 'R']
    record_directory = 'svhn_mnist' if config.source in svhn_mnist and config.target in svhn_mnist \
        else 'mnist_usps' if config.source in mnist_usps and config.target in mnist_usps \
        else 'synsig_gtsrb' if config.source in synsig_gtsrb and config.target in synsig_gtsrb \
        else 'office_31' if config.source in office_31 and config.target in office_31\
        else 'officeHome' if config.source in officeHome and config.target in officeHome else None
    record_directory = f'{record_directory}{"_all_use"}' if record_directory == 'mnist_usps' and config.all_use \
        else f'{record_directory}{"_not_all"}' if record_directory == 'mnist_usps' and not config.all_use \
        else record_directory

    if not record_directory:
        raise ValueError('invalid transfer setting, not in expected transfer domain!')

    record_directory = os.path.join('exp_record', record_directory)
    checkpoint_directory = os.path.join(record_directory, config.checkpoint)
    if not os.path.exists(record_directory):
        os.makedirs(record_directory)
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    return record_directory


def init_record_file_name(config):
    """
    初始化文件存储名
    :param config:
    :return:
    """
    record_directory = init_record_directory(config)
    record_num = 0
    file_name_format_train, file_name_format_test = \
        '{}\\%s_%s_alluse_%s_onestep_%s_record_num_%s_train.txt'.format(record_directory), \
        '{}\\%s_%s_alluse_%s_onestep_%s_record_num_%s_test.txt'.format(record_directory)
    record_train_file_path = file_name_format_train % (
        config.source, config.target, config.all_use, config.one_step, record_num)
    record_test_file_path = file_name_format_test % (
        config.source, config.target, config.all_use, config.one_step, record_num)
    while os.path.exists(record_train_file_path):
        record_num += 1
        record_train_file_path = file_name_format_train % (
            config.source, config.target, config.all_use, config.one_step, record_num)
        record_test_file_path = file_name_format_test % (
            config.source, config.target, config.all_use, config.one_step, record_num)
    return record_directory, record_train_file_path, record_test_file_path


def entropy_loss(output):
    """

    :param output:
    :return:
    """
    return - torch.mean(output * torch.log(output + 1e-6))


def loss_discrepancy(out1, out2):
    """

    :param out1:
    :param out2:
    :return:
    """
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def record_log(record_file_path, info):
    """

    :param record_file_path:
    :param info:
    :return:
    """
    with open(record_file_path, 'a') as f:
        f.write(info)


class GaussianNoise(torch.nn.Module):
    """
    GaussianNoise
    """

    def __init__(self, batch_size, input_shape=(1, 32, 32), std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise


def adr_discrepancy(out1, out2):
    entropy, use_abs_diff = False, False
    if not entropy:
        out2_t = out2.clone()
        out2_t = out2_t.detach()
        out1_t = out1.clone()
        out1_t = out1_t.detach()
        if not use_abs_diff:
            return (F.kl_div(F.log_softmax(out1), out2_t) + F.kl_div(F.log_softmax(out2), out1_t)) / 2
        else:
            return torch.mean(torch.abs(out1-out2))
    else:
        return entropy_loss(out1)


def calculate_grl_coefficient(epoch_num, epoch_total=100.0, high=1.0, low=0.0, alpha=10.0):
    coefficient = np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * epoch_num / float(epoch_total)))
                           - (high - low) + low)
    return coefficient


def update_optimizer(optimizer=None, lr=0.01, epoch_num=0, epoch_total=1, gamma=10, power=0.75, weight_decay=0.0005):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    :param optimizer:
    :param lr:
    :param epoch_num:
    :param gamma:
    :param power:
    :param weight_decay:
    :return:
    """
    # gamma, power = 10, 0.75
    lr = lr * (1 + gamma * (epoch_num / epoch_total)) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']

    return optimizer


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Use the true average until the exponential average is more correct
    :param model:
    :param ema_model:
    :param alpha:
    :param global_step:
    :return:
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def mixup_data_supervised(data, label, alpha=1.0):
    """
    Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
    :param data:
    :param label:
    :param alpha:
    :return:
    """
    if alpha > 0.:
        mix_ratio = np.random.beta(alpha, alpha)
    else:
        mix_ratio = 1.
    batch_size = data.size()[0]
    index = np.random.permutation(batch_size)
    mixed_data = mix_ratio * data + (1 - mix_ratio) * data[index, :]
    label_data_i, lebal_data_j = label, label[index]

    return mixed_data, label_data_i, lebal_data_j, mix_ratio


def mixup_data_unsupervised(data, pseudo_label, alpha=1.0):
    """
    Compute the mixup data. Return mixed inputs, mixed target, and lambda
    :param data:
    :param pseudo_label:
    :param alpha:
    :return:
    """
    if alpha > 0.:
        mix_ratio = np.random.beta(alpha, alpha)
    else:
        mix_ratio = 1.
    batch_size = data.size()[0]
    index = np.random.permutation(batch_size)
    mixed_data = mix_ratio * data + (1 - mix_ratio) * data[index, :]
    mixed_pseudo_label = mix_ratio * pseudo_label + (1 - mix_ratio) * pseudo_label[index, :]

    return mixed_data, mixed_pseudo_label, mix_ratio


def get_loader(domain, method, config):
    """

    :param domain:
    :param method:
    :param config:
    :return:
    """
    office_domain, specific_domain = ['A', 'W', 'D'], domain
    domain = mnist if domain == 'mnist' else usps if domain == 'usps' \
        else office_31 if domain in office_domain else None
    if not domain:
        raise ValueError('domain value error, not in expected datasets!')
    dataloader_train, dataloader_test = None, None
    if method == 'gta':
        dataloader_train, dataloader_test = domain.get_loader_gta(config)
    elif method == 'mcd':
        dataloader_train, dataloader_test = domain.get_loader_mcd(config)
    elif method == 'cogan':
        dataloader_train, dataloader_test = domain.get_loader_cogan(config)
    elif method == 'normal':
        dataloader_train, dataloader_test = domain.get_loader_normal(config=config, domain=specific_domain)

    return dataloader_train, dataloader_test


def filter_data_index(index_list=None, real_label=None, predict_c1=None, predict_c2=None):
    """
    标签过滤
    :param index_list:
    :param real_label:
    :param predict_c1:
    :param predict_c2:
    :return: consistensy_index_dict —> (index, real_label, predict_c1)
            inconsistency_index_dict —> (index, real_label, predict_c1, predict_c2)
    """
    consistensy_index_dict, inconsistency_index_dict = {}, {}
    for index in range(len(index_list)):
        id_value = index_list[index]
        if predict_c1[index] == predict_c2[index]:
            consistensy_index_dict[id_value] = (int(index_list[index]), int(real_label[index]),
                                                int(predict_c1[index]))
        else:
            inconsistency_index_dict[id_value] = (int(index_list[index]), int(real_label[index]),
                                                  int(predict_c1[index]), int(predict_c2[index]))

    return consistensy_index_dict, inconsistency_index_dict




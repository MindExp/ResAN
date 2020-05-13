'''
--source=synsig
--target=gtsrb
--source_loader=mcd
--target_loader=mcd
--num_classes=10
--all_use
--backbone=resnet18
--batch_size=128
--image_size=256
--one_step
--lr=0.001
--max_epoch=200
----optimizer=adam
--ensemble_alpha=0.8
--rampup_length=80
--weight_consistency_upper=1.0
--mixup_beta=0.5
--supplementary_info="target domain mixup consistency loss"
'''

'''
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader


def weightRandomSampler():
    """
    1. 处理类别不平衡问题
    2. 再采样池设计
    :return:
    """
    # Create dummy data with class imbalance 99 to 1
    numDataPoints = 1000
    data_dim = 5
    bs = 100
    data = torch.randn(numDataPoints, data_dim)
    label = torch.cat((torch.zeros(int(numDataPoints * 0.99), dtype=torch.long),
                        torch.ones(int(numDataPoints * 0.01), dtype=torch.long)))

    print('target train 0/1: {}/{}'.format(
        (label == 0).sum(), (label == 1).sum()))

    # Create subset indices
    subset_idx = torch.cat((torch.arange(500), torch.arange(-5, 0)))

    # Compute samples weight (each sample should get its own weight)
    class_sample_count = torch.tensor(
        [(label[subset_idx] == t).sum() for t in torch.unique(label, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[item_label] for item_label in label[subset_idx]])

    # Create sampler, dataset, loader
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    train_dataset = torch.utils.data.TensorDataset(
        data[subset_idx], label[subset_idx])
    train_loader = DataLoader(
        train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

    # Iterate DataLoader and check class balance for each batch
    for i, (x, y) in enumerate(train_loader):
        print("batch index {}, 0/1: {}/{}".format(
            i, (y == 0).sum(), (y == 1).sum()))


if __name__ == '__main__':
    weightRandomSampler()
'''

'''
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class TxtDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self):
        self.data = np.asarray([[100, 101], [103, 104], [2, 1], [6, 4], [4, 5]])  # 特征向量集合,特征是2维表示一段文本
        self.label = np.asarray([[1, ], [2, ], [3, ], [4, ], [5, ]])  # 标签是1维,表示文本类别

    def __getitem__(self, index):
        txt = torch.LongTensor(self.data[index])
        label = torch.LongTensor(self.label[index])
        return index, txt, label  # 返回标签

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    Txt = TxtDataset()
    test_loader = DataLoader(Txt, batch_size=2, shuffle=True,
                             num_workers=0)
    iter_test_loader = iter(test_loader)

    for batch_index in range(len(test_loader)):
        index, data, label = iter_test_loader.next()
        # print('batch_index:', batch_index)
        print('index:', index)
        print('data:', data)
        print('label', label)

'''

"""
import torch
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        print("forward in function!")
        return x.view_as(x)

    def backward(self, grad_output):
        print("backward in function!")
        return -self.lambd * grad_output


def grad_reverse(x, lambd=1.0):
    grl = GradReverse(lambd)
    return grl(x)

def grl_hook(coeff):
    def func(grad):
        print("backward in grl_hook!")
        return -coeff * grad

    return func

import torch.nn as nn

data = torch.Tensor([0, 0, 0])
data.requires_grad = True
# data.register_hook(grl_hook(4.0))
label = torch.Tensor([2, 2, 2])
label.requires_grad = True
_data = 2 * data
_data = grad_reverse(_data, 4.0)
# _data.register_hook(grl_hook(4.0))
loss = nn.MSELoss()
output = loss(_data, label) * len(_data)
output.backward()
print(data.grad.data)
"""

"""
'''
    consistency:
    {'id': (real_c, predict_c)}
    inconsistency:
    {'id': (real_c, predict_c1, predict_c2)}
'''

consistency_dict = {}
inconsistency_dict = {}
label_invisible = [2, 2, 3, 1, 5]
id_data = [21, 33, 65, 48, 67]
predict_c1 = [3, 2, 3, 4, 5]
predict_c2 = [3, 1, 3, 2, 5]

for index in range(len(predict_c2)):
    key = id_data[index]
    if predict_c1[index] != predict_c2[index]:
        inconsistency_dict[key] = (label_invisible[index], predict_c1[index], predict_c2[index])
    else:
        consistency_dict[key] = (label_invisible[index], predict_c1[index])
print(inconsistency_dict)
print(consistency_dict)
inconsistency_dict.update(consistency_dict)
print(inconsistency_dict.keys())
"""

"""
num_labels = len(y_train) if config.num_labels == 'all' else config.num_labels
    if config.corruption_percentage > 0:
        corrupt_labels = int(0.01 * num_labels * config.corruption_percentage)
        corrupt_labels = min(corrupt_labels, num_labels)
        print("Corrupting %d labels." % corrupt_labels)
        for i in range(corrupt_labels):
            y_train[i] = np.random.randint(0, num_classes)
"""
"""
dict_a = {1: 'a', 2: 'b', 3: 'c'}
dict_b = {3: 'fix', 4: 'add'}
# dict_a.update(dict_b)

print(dict_a.__contains__(1))
"""
"""
data_set = set((1, 2, 3, 4, 2))
print(data_set)
filter_data = filter(lambda index: index / 2 == 1, data_set)
result = [i for i in filter_data]
print(result)
"""

"""
import torch
import numpy as np

inconsistency_index = torch.from_numpy(np.array([1, 3, 5, 6], dtype=np.int32))
print(inconsistency_index)
current_inconsistency_index = torch.from_numpy(np.array([1, 2], dtype=np.int32))
print(current_inconsistency_index)
inconsistency_index = np.union1d(inconsistency_index, current_inconsistency_index)
inconsistency_index = np.union1d(inconsistency_index, current_inconsistency_index)
print(inconsistency_index)
"""

"""
from torch.optim import lr_scheduler
lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
"""

# data_set = set((1, 2, 3, 4, 2))
# print(data_set)
# data_new = torch.from_numpy(np.array([1, 2]))
# print(data_new)
# data_new = data_new.numpy()
# data_set.update(data_new)
# print(data_set)
# data_set.update(data_new)
# print(data_set)

"""
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
"""
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

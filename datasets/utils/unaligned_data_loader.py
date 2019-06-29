import torch.utils.data
import torchvision.transforms as transforms

from datasets.utils.datasets import Dataset


class UnalignedDataLoader:
    def __init__(self):
        self.dataloader_train, self.dataloader_test = None, None

    def initialize(self, train_set, test_set, batch_size, scale=32):
        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_train = Dataset(train_set['imgs'], train_set['labels'], transform=transform)
        dataset_test = Dataset(test_set['imgs'], test_set['labels'], transform=transform)
        self.dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
        self.dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.dataloader_train, self.dataloader_test

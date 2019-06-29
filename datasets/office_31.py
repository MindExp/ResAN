import torch
import os
from PIL import Image
import torchvision.transforms as transforms

root_file_path_data_label = 'E:\Projects\dataset\office_31\office_all'
root_file_path_data = 'E:\Projects\dataset\office_31'


class Office(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform

        with open(file_path, "r") as fp:
            content = fp.readlines()
        self.mapping = list(map(lambda x: (os.path.join(root_file_path_data, x[0]), int(x[1])),
                                [content_line.strip().split() for content_line in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        image_name, label = self.mapping[index]
        image = self.transform(Image.open(image_name).convert('RGB'))
        return image, label


def get_loader_normal(config, domain):
    dataset_train, dataset_test = load_office_normal(config, domain)
    dataloader_trainset = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=config.batch_size,
                                                      shuffle=True, drop_last=True)
    dataloader_testset = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=config.batch_size,
                                                     shuffle=False, drop_last=False)

    return dataloader_trainset, dataloader_testset


def load_office_normal(config, domain):
    name_dataset = 'amazon_list.txt' if domain == 'A' else 'webcam_list.txt' if domain == 'W' \
        else 'dslr_list.txt' if domain == 'D' else None

    file_path_data_label = os.path.join(root_file_path_data_label, name_dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_train = Office(file_path=file_path_data_label, transform=transform_train)
    dataset_test = Office(file_path=file_path_data_label, transform=transform_test)

    return dataset_train, dataset_test

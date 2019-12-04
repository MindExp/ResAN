import torch
import torch.nn as nn
import utils.util as util
from models.utils import resnet
from efficientnet_pytorch import EfficientNet
from models.utils.gradient_reverse_layer import grl_hook, grad_reverse

file_path_pretrained_alexnet_model = 'E:\Projects\preTrainedModels\\alexnet.pth.tar'


class LRN(nn.Module):
    """
    Local Response Normalization
    """

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        pretrained_model = torch.load(file_path_pretrained_alexnet_model)
        model.load_state_dict(pretrained_model['state_dict'])
    return model


# TODO 修复 F、C 结构设计
class AlexNet_F(nn.Module):
    def __init__(self):
        super(AlexNet_F, self).__init__()
        model_alexnet = alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.__in_features = model_alexnet.classifier[0].in_features
        # self.classifier = nn.Sequential()
        # for i in range(6):
        #     self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def get_output_features(self):
        return self.__in_features

    def get_parameters(self):
        parameter_list = [{"params": self.features.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class AlexNet_C(nn.Module):
    def __init__(self, netF, use_bottleneck=True, bottleneck_dim=256, new_class=True, class_num=1000):
        super(AlexNet_C, self).__init__()
        self.in_features, self.use_bottleneck, self.new_class = netF.get_output_features(), use_bottleneck, new_class
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # nn.Linear(4096, class_num),
        )
        self.in_features = self.classifier[3].out_features
        if self.new_class:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(self.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                # self.bottleneck.apply(util.init_weights)
                # self.fc.apply(util.init_weights)
            else:
                self.fc = nn.Linear(self.in_features, class_num)
                # self.fc.apply(util.init_weights)
        else:
            self.fc = nn.Linear(self.in_features, class_num)

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)

        x = self.classifier(x)
        if self.use_bottleneck and self.new_class:
            x = self.bottleneck(x)
        y = self.fc(x)
        return y

    def get_parameters(self):
        if self.new_class:
            if self.use_bottleneck:
                parameter_list = [{"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2},
                                  {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
            else:
                parameter_list = [{"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.fc.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


resnet_dict = {"resnet18": resnet.resnet18, "resnet34": resnet.resnet34, "resnet50": resnet.resnet50,
               "resnet101": resnet.resnet101, "resnet152": resnet.resnet152}


# TODO 修复 F、C 结构设计
class ResNet_F(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256):
        super(ResNet_F, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True, online=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.fc = model_resnet.fc

        self.feature_layer = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.avgpool)
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_layer = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.__out_features = bottleneck_dim
        else:
            self.__out_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return x

    def get_output_features(self):
        return self.__out_features

    def get_parameters(self):
        if self.use_bottleneck:
            parameter_list = [{"params": self.feature_layer.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.bottleneck_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.feature_layer.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class ResNet_C(nn.Module):
    def __init__(self, netF, new_class=True, class_num=1000):
        super(ResNet_C, self).__init__()
        self.out_features, self.new_class = netF.get_output_features(), new_class

        if self.new_class:
            mlp_out_units = self.out_features // 2
            self.classifier_layer = nn.Sequential(
                nn.Linear(self.out_features, mlp_out_units),
                nn.BatchNorm1d(mlp_out_units),
                nn.ReLU(inplace=True),

                nn.Linear(mlp_out_units, class_num)
            )

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        if self.new_class:
            x = self.classifier_layer(x)
        else:
            x = self.out_features

        return x

    def get_parameters(self):
        if self.new_class:
            parameter_list = [{"params": self.classifier_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": None, "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class EfficientNet_F(nn.Module):
    def __init__(self, efficientnet_name, new_class=False, use_bottleneck=True, bottleneck_dim=256):
        super(EfficientNet_F, self).__init__()
        self.new_class = new_class
        self.use_bottleneck = use_bottleneck
        self.model_efficientnet = EfficientNet.from_pretrained(efficientnet_name, online=False)

        if self.new_class and self.use_bottleneck:
            self.bottleneck_layer = nn.Linear(self.model_efficientnet._fc.out_features, bottleneck_dim)
            self.__out_features = bottleneck_dim
        else:
            self.__out_features = self.model_efficientnet._fc.out_features

    def forward(self, x):
        # x = self.model.extract_features(x)
        x = self.model_efficientnet(x)
        if self.new_class and self.use_bottleneck:
            x = self.bottleneck_layer(x)
        x = x.view(x.size(0), -1)
        return x

    def get_output_features(self):
        return self.__out_features

    def get_parameters(self):
        if self.use_bottleneck:
            parameter_list = [{"params": self.model_efficientnet.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.bottleneck_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.model_efficientnet.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class EfficientNet_C(nn.Module):
    def __init__(self, netF, new_class=True, class_num=1000):
        super(EfficientNet_C, self).__init__()
        self.out_features, self.new_class = netF.get_output_features(), new_class

        if self.new_class:
            self.classifier_layer = nn.Sequential(
                nn.Linear(self.out_features, self.out_features),
                nn.BatchNorm1d(self.out_features),
                nn.ReLU(inplace=True),

                nn.Linear(self.out_features, class_num)
            )

    def forward(self, x, reverse=False, grl_coefficient=1.0):
        if reverse:
            # x.register_hook(grl_hook(grl_coefficient))
            x = grad_reverse(x, grl_coefficient)
        if self.new_class:
            x = self.classifier_layer(x)
        else:
            x = self.out_features

        return x

    def get_parameters(self):
        if self.new_class:
            parameter_list = [{"params": self.classifier_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": None, "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_params = nn.Sequential(nn.Linear(50 * 4 * 4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(util.init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = x * 1.0
        # method 2: implementation of GRL
        # x.register_hook(grl_hook(grl_coefficient))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    """
    method 1: implementation of GRL
    def backward(self, grad):
        coefficient = util.calculate_grl_coefficient(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        return coefficient * grad
    """

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def get_netF(backbone=None):
    if backbone == 'alexnet':
        netF = AlexNet_F()
    elif 'resnet'in backbone:
        netF = ResNet_F(resnet_name=backbone, use_bottleneck=True, bottleneck_dim=1024)
    elif 'efficientnet'in backbone:
        netF = EfficientNet_F(efficientnet_name=backbone, use_bottleneck=True, bottleneck_dim=256)
    else:
        raise ValueError("Invalid feature extractor base network.")
    return netF


def get_netC(backbone=None, new_class=True, class_num=31):
    if backbone.__class__.__name__ == 'AlexNet_F':
        netC = AlexNet_C(backbone, new_class=new_class, class_num=class_num)
    elif backbone.__class__.__name__ == 'ResNet_F':
        netC = ResNet_C(backbone, new_class=new_class, class_num=class_num)
    elif backbone.__class__.__name__ == 'EfficientNet_F':
        netC = EfficientNet_C(backbone, new_class=new_class, class_num=class_num)
    else:
        raise ValueError("Invalid base network parameter.")
    return netC

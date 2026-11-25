import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from linear_size import linear_size_inputsize

# print(
#     linear_size_inputsize(
#         (224, 224),
#         kernel_size=(3, 2, 3, 2),
#         stride_size=(1, 2, 1, 2),
#         padding_size=(0, 0, 0, 0),
#     )
# )


class ConvolutionaNetwork_CATS_DOGS(nn.Module):
    def __init__(
        self,
        input_size=(224, 224),
        conv_filters_num=[6, 16],
        conv_kernel_size=[3, 3],
        conv_stride_size=[1, 1],
        conv_padding_size=[0, 0],
        pool_kernel_size=[2, 2],
        pool_stride_size=[2, 2],
        pool_padding_size=[0, 0],
        layers=[120, 84],
        out_szs=2,
    ):
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride_size = pool_stride_size
        self.pool_padding_size = pool_padding_size

        kernel_size = []
        kernel_size.extend(*conv_kernel_size)
        kernel_size.extend(*pool_kernel_size)

        stride_size = []
        stride_size.extend(*conv_stride_size)
        stride_size.extend(*pool_stride_size)

        padding_size = []
        padding_size.extend(*conv_padding_size)
        padding_size.extend(*pool_padding_size)
        super().__init__()
        self.conv1 = nn.Conv2d(
            3,
            conv_filters_num[0],
            conv_kernel_size[0],
            conv_stride_size[0],
            conv_padding_size[0],
        )
        self.conv2 = nn.Conv2d(
            conv_filters_num[0],
            conv_filters_num[1],
            conv_kernel_size[1],
            conv_stride_size[1],
            conv_padding_size[1],
        )
        self.fc1 = nn.Linear(
            linear_size_inputsize(
                input_size,
                kernel_size,
                stride_size,
                padding_size,
            )
            * conv_filters_num[-1],
            120,
        )
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_szs)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(
            X,
            self.pool_kernel_size[0],
            self.pool_stride_size[0],
            self.pool_padding_size[0],
        )
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(
            X,
            self.pool_kernel_size[1],
            self.pool_stride_size[1],
            self.pool_padding_size[1],
        )
        X = X.view(len(X), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X


class AlexNetEdition(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Load pretrained AlexNet
        model = models.alexnet(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        # # Modofy classifier
        # for i in (5, 5):
        #     del self.model.classifier[i]
        # self.model.classifier[1] = nn.Linear(9216, 4096)
        # self.model.classifier[4] = nn.Linear(4096, 1024)
        # del self.model.classifier[0]

        # # Add new layers
        # self.model.classifier.add_module("7", nn.ReLU())
        # self.model.classifier.add_module("8", nn.Dropout(dropout))
        # self.model.classifier.add_module("9", nn.Linear(500, num_classes))
        # self.model.classifier.append(nn.Linear(4096, 1024))
        self.classifier = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, X):
        X = self.features(X)
        X = self.avgpool(X)
        X = X.view(len(X), -1)
        X = self.classifier(X)

        return X

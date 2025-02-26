import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes, layers=[18, 34, 50], in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Define the number of blocks for each ResNet variant
        self.block_config = {
            18: [2, 2, 2, 2],  # ResNet-18
            34: [3, 4, 6, 3],  # ResNet-34
            50: [3, 4, 6, 3],  # ResNet-50
        }

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Determine which ResNet version is selected (18, 34, or 50)
        self.layers = layers[0]  # Select the first ResNet from the list

        self.layer1 = self._make_layer(64, 128, self.block_config[self.layers][0], stride=1)
        self.layer2 = self._make_layer(128, 256, self.block_config[self.layers][1], stride=2)
        self.layer3 = self._make_layer(256, 512, self.block_config[self.layers][2], stride=2)
        self.layer4 = self._make_layer(512, 1024, self.block_config[self.layers][3], stride=2)

        self.fc = nn.Linear(1024, num_classes)  # Final fully connected layer

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.mean([2, 3])  # Global Average Pooling
        x = self.fc(x)
        return x

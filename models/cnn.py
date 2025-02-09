import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional neural network model definition
    """

    def __init__(self, in_channels: int, num_classes: int):
        """
        Creates an instance of the `CNN` class

        Parameters:
            in_channels (int): Number of channels in input image
            num_classes (int): Number of classes to predict
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.in_channels, 6, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass for the `CNN` model

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, num_classes)
        """
        x = self.relu(self.bn1(self.conv1(x)))  # (1, 28, 28) -> (6, 24, 24)
        x = self.maxpool(x)  # (6, 24, 24) -> (6, 12, 12)
        x = self.relu(self.bn2(self.conv2(x)))  # (6, 12, 12) -> (16, 8, 8)
        x = self.maxpool(x)  # (16, 8, 8) -> (16, 4, 4)

        x = self.flatten(x)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x

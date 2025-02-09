import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron model definition
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Creates an instance of the `MLP` class

        Parameters:
            input_dim (int): Dimensionality of input tensor
            output_dim (int): Number of output classes to predict
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the `MLP` model

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W)

        Returns:
            (torch.Tensor): Output tensor of shape (B, num_classes)
        """
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

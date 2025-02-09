from typing import Callable, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import optim
from torch.utils.data import DataLoader, Dataset


def train_model(
    model: Callable,
    train_set: Dataset,
    valid_set: Dataset,
    loss_fn: Callable,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 0.001,
    optimizer_cls=optim.Adam,
    device: str = "cpu",
) -> Tuple[Callable, ArrayLike, ArrayLike]:
    """
    Train a PyTorch model with specified hyperparameters

    Parameters:
        model: Input model to train
        train_set: PyTorch Dataset object with training data
        valid_set:PyTorch Dataset object with validation data
        loss_fn: Loss function used to calculate loss
        batch_size: Number of data points to process in a batch
        num_epochs: Number of epochs to train the model for
        lr: Learning rate used with the input optimizer
        optimizer: Optimizer used for automatic weight updates
        device: Device to train the model on

    Returns:
        (Tuple[Callable, ArrayLike, ArrayLike]): A tuple of the trained model, the
            training loss history, and the validation loss history
    """

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # Send model to device
    model.to(device)

    # Set up optimizer to work with input model
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    # Store loss metrics
    train_loss_history = np.zeros([num_epochs, 1])
    valid_loss_history = np.zeros([num_epochs, 1])

    for epoch in range(num_epochs):

        # Put model into train mode
        model.train()

        running_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            # Send data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            pred = model(data)

            # Calculate loss
            loss = loss_fn(pred, target)
            running_loss += loss.item()

            # Backward pass
            loss.backward()

            # Step with optimizer
            optimizer.step()

        train_loss_history[epoch] = running_loss / len(train_loader.dataset)

        print(f"Train Epoch {epoch + 1}: Average Loss {train_loss_history[epoch]:.4f} ")

        # Put model into evaluation mode
        model.eval()

        valid_loss = 0

        # Turn off autograd
        with torch.no_grad():
            for data, target in valid_loader:

                # Send to device
                data, target = data.to(device), target.to(device)

                # Forward pass
                pred = model(data)

                valid_loss += loss_fn(pred, target).item()

            valid_loss_history[epoch] = valid_loss / len(valid_loader.dataset)

            print(
                f"Valid Epoch {epoch + 1}: Average Loss {valid_loss_history[epoch]:.4f} "
            )

    return model, train_loss_history, valid_loss_history


def test_model(
    model: Callable,
    test_set: Dataset,
    loss_fn: Callable,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Evalute a PyTorch model using an input test set

    Parameters:
        model: PyTorch model to evaluate
        test_set: Test dataset used to evaluate the input model
        loss_fn: Loss function used to calculate loss
        batch_size: Number of data points to process in a batch
        device: Device to test the model on

    Returns:
        (float): The loss of the input model based on the input loss function
    """
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Put model into evaluation mode
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:

            # Send inputs to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            pred = model(data)

            test_loss += loss_fn(pred, target).item()

    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}")

    return test_loss

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SuccessorFeatureNetwork(nn.Module):
    def __init__(self, input_dim: int = 63, output_dim: int = 63):
        super().__init__()

        # Architecture: input -> 128 -> 128 -> 128 -> 128
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else 128, 128),
                    nn.ReLU(),
                )
                for i in range(3)
            ]
        )

        # Final linear layer outputs successor features
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class SuccessorFeatureNetworkDeep(nn.Module):
    def __init__(self, input_dim: int = 63, output_dim: int | None = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.layer1 = nn.Linear(input_dim, input_dim)
        self.layer2 = nn.Linear(input_dim, input_dim)
        self.layer3 = nn.Linear(input_dim, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(s))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class SuccessorFeatureModel(nn.Module):
    def __init__(self, state_dim: int = 63):
        super().__init__()
        self.network = SuccessorFeatureNetwork(
            input_dim=state_dim, output_dim=state_dim
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.network(s)


class SuccessorFeatureModelDeep(nn.Module):
    def __init__(self, state_dim: int = 63):
        super().__init__()
        self.network = SuccessorFeatureNetworkDeep(
            input_dim=state_dim, output_dim=state_dim
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.network(s)


class StateTransitionDataset(Dataset):
    """
    Dataset for state transitions (s, s') from a single policy.

    Args:
        transitions: List of (s, s') pairs where each is a numpy array
    """

    def __init__(self, transitions: List[Tuple[np.ndarray, np.ndarray]]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s, s_next = self.transitions[idx]
        return torch.from_numpy(s).float(), torch.from_numpy(s_next).float()


def bellman_loss(
    model: SuccessorFeatureModel,
    s: torch.Tensor,
    s_next: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute the Bellman loss for successor features.

    Loss: ||ψ_θ^π(s) - (φ_θ(s) + γ * ψ_θ^π(s'))||²₂

    Args:
        model: Successor feature model
        s: Current states tensor of shape (batch_size, state_dim)
        s_next: Next states tensor of shape (batch_size, state_dim)
        gamma: Discount factor (same as in the SARSA algorithm)

    Returns:
        Mean squared Bellman loss
    """
    phi_s = s

    with torch.no_grad():
        psi_s_next = model(s_next)

    # Compute target: φ_θ(s) + γ * ψ_θ^π(s')
    target = phi_s + gamma * psi_s_next

    # Compute MSE loss: ||ψ_θ^π(s) - (φ_θ(s) + γ * ψ_θ^π(s'))||²₂
    loss = F.mse_loss(model(s), target)

    return loss


def train_epoch(
    model: SuccessorFeatureModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    gamma: float = 0.99,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: Successor feature model
        dataloader: DataLoader for state transitions
        optimizer: Optimizer for model parameters
        gamma: Discount factor
        device: Device to run training on

    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, (s, s_next) in enumerate(dataloader):
        s = s.to(device)
        s_next = s_next.to(device)

        # Forward pass and compute loss
        loss = bellman_loss(model, s, s_next, gamma)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def create_dataloader(
    transitions: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for state transitions.

    Args:
        transitions: List of (s, s') pairs
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader for state transitions
    """
    dataset = StateTransitionDataset(transitions)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def save_model(model: SuccessorFeatureModel, policy_name: str) -> Path:
    model_dir = Path("psi_models")
    model_dir.mkdir(exist_ok=True)

    # Create model file path
    model_path = model_dir / f"{policy_name}.pth"

    # Save the model state dict
    torch.save(model.state_dict(), model_path)

    return model_path

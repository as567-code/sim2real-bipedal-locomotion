from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class RunningMeanStd(nn.Module):
    """Welford online running mean and standard deviation.

    Used for observation normalization. Tracks count, mean, and variance
    incrementally. Can be saved/loaded as part of a PyTorch state_dict.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip = clip

        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats with a batch of observations."""
        x = x.cpu().double()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics.

        Handles cross-device tensors: stats live on CPU (float64),
        normalization happens in float32, result returns to input device.
        """
        target_device = x.device
        x_cpu = x.float().cpu()
        mean = self.mean.float()
        std = torch.sqrt(self.var.float() + self.epsilon)
        normed = torch.clamp((x_cpu - mean) / std, -self.clip, self.clip)
        return normed.to(target_device)

    def to_numpy_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Export mean and std as numpy arrays (for ONNX baking)."""
        mean = self.mean.cpu().numpy().astype(np.float32)
        std = np.sqrt(self.var.cpu().numpy().astype(np.float64) + self.epsilon).astype(np.float32)
        return mean, std

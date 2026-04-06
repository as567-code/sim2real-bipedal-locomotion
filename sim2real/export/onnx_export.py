"""ONNX export: bakes observation normalizer into actor network for single-graph inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sim2real.algo.actor_critic import ActorCritic
from sim2real.algo.normalizer import RunningMeanStd


class NormalizedActor(nn.Module):
    """Wraps normalizer + actor mean + same action bounds as training.

    Matches deterministic inference: Gaussian mean from the actor MLP (no tanh
    squashing), then clamp to [-1, 1] like ``BipedalEnv.step``. Training never
    applied tanh to the policy mean; tanh here would skew the deployed policy.
    """

    def __init__(self, actor_mean: nn.Module, obs_mean: torch.Tensor, obs_std: torch.Tensor):
        super().__init__()
        self.actor_mean = actor_mean
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.obs_mean) / self.obs_std
        normalized = torch.clamp(normalized, -10.0, 10.0)
        return torch.clamp(self.actor_mean(normalized), -1.0, 1.0)


def export_to_onnx(
    policy: ActorCritic,
    normalizer: RunningMeanStd,
    output_path: str | Path,
    obs_dim: int = 32,
    opset_version: int = 17,
    validate: bool = True,
) -> Path:
    """Export trained policy to ONNX with baked normalization.

    Args:
        policy: trained ActorCritic module
        normalizer: fitted RunningMeanStd
        output_path: .onnx file path
        obs_dim: observation dimension
        opset_version: ONNX opset
        validate: if True, verify ONNX output matches PyTorch

    Returns:
        Path to saved ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mean_np, std_np = normalizer.to_numpy_params()
    obs_mean = torch.from_numpy(mean_np)
    obs_std = torch.from_numpy(std_np)

    normalized_actor = NormalizedActor(policy.actor_mean, obs_mean, obs_std)
    normalized_actor.train(False)

    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        normalized_actor,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to {output_path}")

    if validate:
        _validate_onnx(normalized_actor, output_path, obs_dim)

    return output_path


def _validate_onnx(torch_model: nn.Module, onnx_path: Path, obs_dim: int) -> None:
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model passed validation check")

    session = ort.InferenceSession(str(onnx_path))

    test_inputs = [
        np.random.randn(1, obs_dim).astype(np.float32),
        np.random.randn(8, obs_dim).astype(np.float32),
        np.zeros((1, obs_dim), dtype=np.float32),
    ]

    for i, test_input in enumerate(test_inputs):
        torch_input = torch.from_numpy(test_input)
        with torch.no_grad():
            torch_output = torch_model(torch_input).numpy()

        ort_output = session.run(None, {"observation": test_input})[0]

        max_diff = np.max(np.abs(torch_output - ort_output))
        assert max_diff < 1e-5, f"Validation failed on test {i}: max_diff={max_diff}"
        print(f"  Test {i}: max_diff={max_diff:.2e} (PASS)")

    print("All ONNX validation tests passed")

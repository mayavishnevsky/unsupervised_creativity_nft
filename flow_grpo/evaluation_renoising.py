"""Deterministic evaluation-only re-noising for FlowMatch samples."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch


RENOISING_SEED_DOMAIN = "nft-evaluation-renoising-v1"


def renoising_seed(
    sample_seed: int,
    step_index: int,
    repeat_index: int,
) -> int:
    """Derive a stable seed independent of evaluation batching and rank."""

    payload = (
        f"{RENOISING_SEED_DOMAIN}|{sample_seed}|{step_index}|{repeat_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        2**63 - 1
    )


def flow_forward_renoise(
    lower_noise_latents: torch.Tensor,
    iid_noise: torch.Tensor,
    sigma_current: float,
    sigma_next: float,
) -> torch.Tensor:
    """Move a FlowMatch sample from sigma_next back to sigma_current."""

    sigma_current = float(sigma_current)
    sigma_next = float(sigma_next)
    if not 0.0 <= sigma_next <= sigma_current <= 1.0 + 1e-6:
        raise ValueError(
            "FlowMatch re-noising requires 0 <= sigma_next <= "
            f"sigma_current <= 1; got {sigma_next} -> {sigma_current}"
        )
    alpha_current = 1.0 - sigma_current
    alpha_next = 1.0 - sigma_next
    if alpha_next <= 0.0:
        raise ValueError("sigma_next must be less than 1 for FlowMatch re-noising")

    carry = alpha_current / alpha_next
    variance = sigma_current**2 - (carry * sigma_next) ** 2
    if variance < -1e-6:
        raise ValueError(
            "invalid FlowMatch re-noising variance for sigmas "
            f"{sigma_current} -> {sigma_next}: {variance}"
        )
    return lower_noise_latents * carry + iid_noise * math.sqrt(max(variance, 0.0))


class EvaluationRenoising:
    """Provide deterministic re-noised latents for selected creative steps."""

    def __init__(
        self,
        *,
        sample_seeds: Sequence[int],
        active_steps: Sequence[bool],
        repeats: int,
    ):
        if isinstance(repeats, bool) or int(repeats) != repeats or repeats <= 0:
            raise ValueError("renoising repeats must be a positive integer")
        if not sample_seeds:
            raise ValueError("renoising requires at least one sample seed")
        if not active_steps:
            raise ValueError("renoising requires at least one inference step")
        self.sample_seeds = tuple(int(seed) for seed in sample_seeds)
        self.active_steps = tuple(bool(active) for active in active_steps)
        self.repeats = int(repeats)

    def validate_step_count(self, step_count: int) -> None:
        if len(self.active_steps) != int(step_count):
            raise ValueError(
                "renoising active-step count must equal sampler step count: "
                f"{len(self.active_steps)} != {step_count}"
            )

    def is_active(self, step_index: int) -> bool:
        return self.active_steps[step_index]

    def _iid_noise(
        self,
        latents: torch.Tensor,
        step_index: int,
        repeat_index: int,
    ) -> torch.Tensor:
        if latents.shape[0] != len(self.sample_seeds):
            raise ValueError(
                "renoising seed count must equal latent batch size: "
                f"{len(self.sample_seeds)} != {latents.shape[0]}"
            )
        rows = []
        for sample_seed in self.sample_seeds:
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(
                renoising_seed(sample_seed, step_index, repeat_index)
            )
            rows.append(
                torch.randn(
                    latents.shape[1:],
                    generator=generator,
                    device=latents.device,
                    dtype=latents.dtype,
                )
            )
        return torch.stack(rows)

    def renoise(
        self,
        lower_noise_latents: torch.Tensor,
        sigma_current: torch.Tensor,
        sigma_next: torch.Tensor,
        step_index: int,
        repeat_index: int,
    ) -> torch.Tensor:
        return flow_forward_renoise(
            lower_noise_latents,
            self._iid_noise(lower_noise_latents, step_index, repeat_index),
            float(sigma_current),
            float(sigma_next),
        )

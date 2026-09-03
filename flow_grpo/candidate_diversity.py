"""Inference-time diversity controls for NFT training candidates.

The controls in this module are intentionally scoped to the policy candidate
pipeline call. Reference generation, validation, creative probes, reward
conditioning, and the clean prompt embeddings saved for NFT optimization do
not pass through these controls.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


@dataclass(frozen=True)
class CADSConfig:
    """Condition-Annealed Diffusion Sampler parameters."""

    tau1: float = 0.6
    tau2: float = 0.9
    noise_scale: float = 0.25
    psi: float = 0.5
    rescale: bool = True


@dataclass(frozen=True)
class ContextualRepulsionConfig:
    """Contextual-repulsion parameters used by the SD3.5 implementation."""

    target_tensor: str = "text"
    t_stop: int = 4
    num_steps: int = 100
    eta: float = 6.0e5
    kernel_reg: float = 1.0e-4


def _value(config, name, default):
    if config is None:
        return default
    return getattr(config, name, default)


def resolve_candidate_diversity(
    config,
) -> tuple[str, CADSConfig | ContextualRepulsionConfig | None]:
    """Resolve and validate the optional candidate-diversity configuration."""

    diversity = getattr(config, "candidate_diversity", None)
    method = str(_value(diversity, "method", "none")).lower()
    if method == "none":
        return method, None
    if method not in {"cads", "contextual_repulsion"}:
        raise ValueError(
            "candidate_diversity.method must be one of: "
            f"none, cads, contextual_repulsion; got {method!r}"
        )

    if method == "cads":
        raw = _value(diversity, "cads", None)
        resolved = CADSConfig(
            tau1=float(_value(raw, "tau1", 0.6)),
            tau2=float(_value(raw, "tau2", 0.9)),
            noise_scale=float(_value(raw, "noise_scale", 0.25)),
            psi=float(_value(raw, "psi", 0.5)),
            rescale=bool(_value(raw, "rescale", True)),
        )
        if not 0.0 <= resolved.tau1 < resolved.tau2 <= 1.0:
            raise ValueError("CADS requires 0 <= tau1 < tau2 <= 1")
        if not math.isfinite(resolved.noise_scale) or resolved.noise_scale < 0:
            raise ValueError("CADS noise_scale must be finite and nonnegative")
        if not math.isfinite(resolved.psi) or not 0.0 <= resolved.psi <= 1.0:
            raise ValueError("CADS psi must be finite and lie in [0, 1]")
        return method, resolved

    raw = _value(diversity, "contextual_repulsion", None)
    resolved = ContextualRepulsionConfig(
        target_tensor=str(_value(raw, "target_tensor", "text")).lower(),
        t_stop=int(_value(raw, "t_stop", 4)),
        num_steps=int(_value(raw, "num_steps", 100)),
        eta=float(_value(raw, "eta", 6.0e5)),
        kernel_reg=float(_value(raw, "kernel_reg", 1.0e-4)),
    )
    if resolved.target_tensor not in {"text", "image", "both"}:
        raise ValueError(
            "contextual_repulsion.target_tensor must be text, image, or both"
        )
    if resolved.t_stop < 0:
        raise ValueError("contextual_repulsion.t_stop must be nonnegative")
    if resolved.num_steps <= 0:
        raise ValueError("contextual_repulsion.num_steps must be positive")
    if not math.isfinite(resolved.eta) or resolved.eta < 0:
        raise ValueError("contextual_repulsion.eta must be finite and nonnegative")
    if not math.isfinite(resolved.kernel_reg) or resolved.kernel_reg < 0:
        raise ValueError(
            "contextual_repulsion.kernel_reg must be finite and nonnegative"
        )
    return method, resolved


def validate_candidate_diversity_sampling(config):
    """Validate constraints imposed by NFT's same-prompt batch sampler."""

    method, method_config = resolve_candidate_diversity(config)
    if method == "none":
        return method, method_config

    if float(config.sample.guidance_scale) != 1.0:
        raise ValueError(
            "NFT candidate diversity requires sample.guidance_scale=1; the "
            "CFG-distilled baseline is sampled without a CFG batch"
        )

    if method == "contextual_repulsion":
        sampling_batch_size = int(config.sample.train_batch_size)
        candidates_per_prompt = int(config.sample.num_image_per_prompt)
        if sampling_batch_size != candidates_per_prompt:
            raise ValueError(
                "contextual_repulsion requires sample.train_batch_size to "
                "equal sample.num_image_per_prompt, so one attention call "
                "contains the complete same-prompt candidate group"
            )
        if candidates_per_prompt < 2:
            raise ValueError(
                "contextual_repulsion requires at least two candidates per prompt"
            )

    return method, method_config


def candidate_diversity_seed(
    base_seed: int,
    epoch: int,
    rank: int,
    batch_start: int,
) -> int:
    """Derive a stable CADS seed without advancing NFT's sampling RNG."""

    identity = (
        f"nft-candidate-diversity:{base_seed}:{epoch}:{rank}:{batch_start}"
    ).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (
        2**63 - 1
    )


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def cads_gamma(normalized_timestep: float, tau1: float, tau2: float) -> float:
    """Return the piecewise-linear CADS signal coefficient."""

    timestep = min(max(float(normalized_timestep), 0.0), 1.0)
    if timestep <= tau1:
        return 1.0
    if timestep >= tau2:
        return 0.0
    return (tau2 - timestep) / (tau2 - tau1)


def corrupt_condition(
    clean_condition: torch.Tensor,
    gamma: float,
    config: CADSConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the CADS corruption and optional moment rescaling."""

    gamma = min(max(float(gamma), 0.0), 1.0)
    if gamma == 1.0:
        return clean_condition

    noise = torch.randn(
        clean_condition.shape,
        generator=generator,
        device=clean_condition.device,
        dtype=clean_condition.dtype,
    )
    corrupted = (
        math.sqrt(gamma) * clean_condition
        + config.noise_scale * math.sqrt(1.0 - gamma) * noise
    )
    if not config.rescale or config.psi == 0.0:
        return corrupted

    reduce_dims = tuple(range(1, clean_condition.ndim))
    clean_float = clean_condition.float()
    corrupted_float = corrupted.float()
    clean_mean = clean_float.mean(dim=reduce_dims, keepdim=True)
    clean_std = clean_float.std(dim=reduce_dims, keepdim=True, unbiased=False)
    corrupted_mean = corrupted_float.mean(dim=reduce_dims, keepdim=True)
    corrupted_std = corrupted_float.std(
        dim=reduce_dims,
        keepdim=True,
        unbiased=False,
    ).clamp_min(torch.finfo(torch.float32).eps)
    rescaled = (
        (corrupted_float - corrupted_mean) / corrupted_std * clean_std
        + clean_mean
    )
    return (
        config.psi * rescaled + (1.0 - config.psi) * corrupted_float
    ).to(clean_condition.dtype)


class CADSConditioning:
    """Return noisy prompt tensors for one SD3 flow-matching timestep."""

    def __init__(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        config: CADSConfig,
        generator: torch.Generator,
    ):
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.config = config
        self.generator = generator

    def __call__(self, normalized_timestep: float):
        gamma = cads_gamma(
            normalized_timestep,
            self.config.tau1,
            self.config.tau2,
        )
        return (
            corrupt_condition(
                self.prompt_embeds,
                gamma,
                self.config,
                self.generator,
            ),
            corrupt_condition(
                self.pooled_prompt_embeds,
                gamma,
                self.config,
                self.generator,
            ),
        )


def cosine_vendi_gradient(
    tensor: torch.Tensor,
    kernel_reg: float,
) -> torch.Tensor:
    """Return the flattened cosine log-Vendi gradient."""

    original_shape = tensor.shape
    batch_size = tensor.shape[0]
    flattened = tensor.reshape(batch_size, -1)
    norms = flattened.norm(dim=1, keepdim=True).clamp(min=1.0e-12)
    normalized = flattened / norms
    kernel = normalized @ normalized.T
    kernel = kernel + kernel_reg * torch.eye(
        batch_size,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    trace = kernel.diag().sum().clamp(min=1.0e-12)
    density_matrix = kernel / trace
    eig_input = (
        density_matrix.float()
        if density_matrix.dtype in {torch.bfloat16, torch.float16}
        else density_matrix
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(eig_input)
    eigenvalues = eigenvalues.clamp(min=1.0e-12).to(tensor.dtype)
    eigenvectors = eigenvectors.to(tensor.dtype)

    score_gradient = eigenvectors @ torch.diag(
        -eigenvalues.log() - 1.0
    ) @ eigenvectors.T
    trace_correction = (score_gradient * density_matrix).sum()
    kernel_gradient = (
        score_gradient
        - trace_correction
        * torch.eye(
            batch_size,
            device=tensor.device,
            dtype=tensor.dtype,
        )
    ) / trace
    normalized_gradient = (kernel_gradient + kernel_gradient.T) @ normalized
    tangent_component = (
        normalized_gradient * normalized
    ).sum(dim=1, keepdim=True)
    input_gradient = (
        normalized_gradient - tangent_component * normalized
    ) / norms
    return input_gradient.reshape(original_shape)


class ContextualRepulsionAlgorithm:
    def __init__(self, config: ContextualRepulsionConfig):
        self.config = config

    def _transform(self, tensor: torch.Tensor) -> torch.Tensor:
        step_size = self.config.eta / self.config.num_steps
        for _ in range(self.config.num_steps):
            tensor = tensor + step_size * cosine_vendi_gradient(
                tensor,
                self.config.kernel_reg,
            )
        return tensor

    def apply(self, output, denoising_step: int):
        if denoising_step >= self.config.t_stop:
            return output
        if isinstance(output, tuple):
            image_output, text_output = output
            if self.config.target_tensor in {"image", "both"}:
                image_output = self._transform(image_output)
            if text_output is not None and self.config.target_tensor in {
                "text",
                "both",
            }:
                text_output = self._transform(text_output)
            return image_output, text_output
        if self.config.target_tensor in {"image", "both"}:
            return self._transform(output)
        return output


class ContextualRepulsionState:
    def __init__(self, processors_per_step: int):
        if processors_per_step <= 0:
            raise ValueError("contextual repulsion found no attention processors")
        self.processors_per_step = processors_per_step
        self.call_count = 0

    @property
    def denoising_step(self) -> int:
        return self.call_count // self.processors_per_step


class ContextualRepulsionProcessor:
    def __init__(self, inner, algorithm, state):
        self.inner = inner
        self.algorithm = algorithm
        self.state = state

    def __call__(self, *args, **kwargs):
        output = self.inner(*args, **kwargs)
        output = self.algorithm.apply(output, self.state.denoising_step)
        self.state.call_count += 1
        return output

    def __getattr__(self, name):
        return getattr(self.inner, name)


@contextmanager
def activate_contextual_repulsion(
    pipe,
    config: ContextualRepulsionConfig,
) -> Iterator[ContextualRepulsionState]:
    """Temporarily wrap attention processors for one candidate generation."""

    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not hasattr(transformer, "attn_processors"):
        raise ValueError(
            "contextual_repulsion requires a transformer with Diffusers "
            "attention processors"
        )
    original_processors = dict(transformer.attn_processors)
    state = ContextualRepulsionState(len(original_processors))
    algorithm = ContextualRepulsionAlgorithm(config)
    wrapped_processors = {
        name: ContextualRepulsionProcessor(processor, algorithm, state)
        for name, processor in original_processors.items()
    }
    try:
        transformer.set_attn_processor(dict(wrapped_processors))
        yield state
    finally:
        transformer.set_attn_processor(dict(original_processors))


@contextmanager
def candidate_diversity_controls(
    pipe,
    method: str,
    method_config: CADSConfig | ContextualRepulsionConfig | None,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompts: Sequence[str],
    generator: torch.Generator | None,
) -> Iterator[dict]:
    """Yield kwargs/context active only around a policy candidate call."""

    if method == "none":
        yield {}
        return
    if method == "cads":
        if generator is None:
            raise ValueError("CADS requires a dedicated generator")
        yield {
            "conditioning_step_callback": CADSConditioning(
                prompt_embeds,
                pooled_prompt_embeds,
                method_config,
                generator,
            )
        }
        return
    if method != "contextual_repulsion":
        raise ValueError(f"unsupported candidate diversity method: {method}")
    if len(set(prompts)) != 1:
        raise ValueError(
            "contextual_repulsion requires one same-prompt candidate batch"
        )
    with activate_contextual_repulsion(pipe, method_config):
        yield {}

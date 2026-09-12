"""Conditional latent IEM reward for Stable Diffusion 3.

The notation follows ``Unsupervised Creative Generation``. For every endpoint
``x_0`` and SNR boundary ``gamma``, the frozen baseline supplies the clean
prediction ``f`` used by the denoising residual in equation 14. The baseline
may include a merged LoRA, but excludes RAM's trainable and lagged adapters.
Equation 16 turns those residuals into a finite-dimensional feature map, and
equation 21 evaluates a candidate against a reference cloud through its mean
feature and mean squared feature norm.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from flow_grpo.reference_cache import (
    CLIP_KEY,
    DINO_KEY,
    IEM_MEAN_KEY,
    IEM_NOISE_SEED_KEY,
    IEM_NOISE_SHA256_KEY,
    IEM_VARIANCE_KEY,
    LATENT_KEY,
    ReferenceCacheReader,
)
from flow_grpo.reference_model_schedule import (
    ReferenceModelUpdatePolicy,
    reference_adapter_context,
)


Tensor = torch.Tensor

IMAGE_DISTANCE_MODEL_FIELDS = {
    "clip_cosine": "clip_model_id",
    "clip_l2": "clip_model_id",
    "dino_cosine": "dino_model_id",
    "dino_l2": "dino_model_id",
    "tpips_overall": "tpips_model_id",
}
IEM_FEATURE_WEIGHTING = "sqrt_delta_gamma_v1"
IEM_NOISE_ASSIGNMENT = "rank_permutation_v1"
IEM_OBJECTIVES = (
    "expected_squared_distance",
    "negative_g",
    "negative_log_p_plus_negative_g",
)
REFERENCE_MEAN_IEM_OBJECTIVES = frozenset(
    ("negative_g", "negative_log_p_plus_negative_g")
)
IEM_ASSIGNMENT_SEED_OFFSET = 60_000_000
IEM_NOISE_SEED_OFFSET = 90_000_000
SUPPORTED_DISTANCE_METRICS = ("iem", *IMAGE_DISTANCE_MODEL_FIELDS)
L2_DISTANCE_METRICS = frozenset(("clip_l2", "dino_l2"))


class TorchDistributedContext:
    """Minimal distributed and autocast API consumed by IEMReward."""

    def __init__(self, device, mixed_precision_dtype=None):
        self.device = torch.device(device)
        self.mixed_precision_dtype = mixed_precision_dtype
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.num_processes = torch.distributed.get_world_size() if distributed else 1
        self.process_index = torch.distributed.get_rank() if distributed else 0

    def autocast(self):
        if self.mixed_precision_dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.mixed_precision_dtype)

    def reduce(self, tensor: Tensor, reduction: str = "sum") -> Tensor:
        if self.num_processes == 1:
            return tensor
        result = tensor.clone()
        op = {
            "sum": torch.distributed.ReduceOp.SUM,
            "mean": torch.distributed.ReduceOp.AVG,
        }.get(reduction)
        if op is None:
            raise ValueError(f"unsupported distributed reduction: {reduction}")
        torch.distributed.all_reduce(result, op=op)
        return result

    def gather(self, tensor: Tensor) -> Tensor:
        if self.num_processes == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.num_processes)]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)


def synchronized_time(device) -> float:
    """Return a wall-clock timestamp after pending CUDA work has completed."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@dataclass(frozen=True)
class PromptRecord:
    """A prompt and its stable row id in a collection of source files."""

    prompt_id: int
    text: str


class BalancedPromptSampler:
    """Sample equal-sized, deterministic quotas from configurable text files."""

    def __init__(self, paths: Sequence[str | Path]):
        if not paths:
            raise ValueError("at least one prompt source is required")

        self.paths = tuple(Path(path).resolve() for path in paths)
        self._sources: list[list[PromptRecord]] = []
        self._records: list[PromptRecord] = []
        digest = hashlib.sha256()

        for source_index, path in enumerate(self.paths):
            if not path.is_file():
                raise FileNotFoundError(f"prompt source does not exist: {path}")
            contents = path.read_bytes()
            digest.update(source_index.to_bytes(8, byteorder="little"))
            digest.update(contents)
            lines = [line.strip() for line in contents.decode("utf-8").splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"prompt source is empty: {path}")

            source = []
            for text in lines:
                record = PromptRecord(len(self._records), text)
                self._records.append(record)
                source.append(record)
            self._sources.append(source)

        # A no-repeat cycle treats prompt text case-insensitively across every
        # source. The first source containing a duplicate owns that prompt.
        seen_cycle_prompts: set[str] = set()
        self._cycle_sources: list[list[PromptRecord]] = []
        for source in self._sources:
            unique_source = []
            for record in source:
                key = record.text.casefold()
                if key in seen_cycle_prompts:
                    continue
                seen_cycle_prompts.add(key)
                unique_source.append(record)
            self._cycle_sources.append(unique_source)
        self.source_sha256 = digest.hexdigest()

    def __len__(self) -> int:
        return len(self._records)

    def prompt_for_id(self, prompt_id: int) -> str:
        prompt_id = int(prompt_id)
        if prompt_id < 0 or prompt_id >= len(self._records):
            raise ValueError(f"prompt id is out of range: {prompt_id}")
        return self._records[prompt_id].text

    def _validated_source_weights(
        self,
        source_weights: Sequence[float] | None,
    ) -> tuple[float, ...]:
        if source_weights is None:
            return (1.0,) * len(self._sources)
        if isinstance(source_weights, (str, bytes)):
            raise ValueError("candidate prompt source weights must be a sequence")
        weights = tuple(float(value) for value in source_weights)
        if len(weights) != len(self._sources):
            raise ValueError(
                "candidate prompt source weights must match candidate_prompt_files: "
                f"got {len(weights)} weights for {len(self._sources)} files"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("candidate prompt source weights must be finite and positive")
        return weights

    def _no_repeat_source_order(
        self,
        seed: int,
        source_index: int,
    ) -> list[PromptRecord]:
        """Return the fixed shuffled order repeated by one prompt source."""

        source = self._cycle_sources[source_index]
        if not source:
            raise ValueError(
                f"prompt source {self.paths[source_index]} has no unique prompts"
            )
        payload = (
            f"prompt-source-order\0{int(seed)}\0{source_index}\0"
            f"{self.source_sha256}"
        ).encode("utf-8")
        order_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        order = list(source)
        random.Random(order_seed).shuffle(order)
        return order

    def sample_no_repeat_cycle(
        self,
        count: int,
        seed: int,
        epoch: int,
        source_weights: Sequence[float] | None = None,
    ) -> list[PromptRecord]:
        """Keep source ratios fixed and cycle each source independently."""

        count = int(count)
        epoch = int(epoch)
        if count < 1:
            raise ValueError(f"prompt sample count must be positive, got {count}")
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")

        weights = self._validated_source_weights(source_weights)
        source_orders = [
            self._no_repeat_source_order(seed, source_index)
            for source_index in range(len(self._cycle_sources))
        ]
        total_weight = sum(weights)
        absolute_start = epoch * count
        absolute_stop = absolute_start + count
        credits = [0.0] * len(source_orders)
        source_counts = [0] * len(source_orders)
        schedule_payload = (
            f"prompt-source-schedule\0{int(seed)}\0{self.source_sha256}"
        ).encode("utf-8")
        schedule_seed = int.from_bytes(
            hashlib.sha256(schedule_payload).digest()[:8],
            "little",
        )
        source_order = list(range(len(source_orders)))
        random.Random(schedule_seed).shuffle(source_order)
        tie_rank = {source_index: rank for rank, source_index in enumerate(source_order)}
        selected: list[PromptRecord] = []

        for position in range(absolute_stop):
            for index in range(len(source_orders)):
                credits[index] += weights[index]
            selected_source = max(
                range(len(source_orders)),
                key=lambda index: (credits[index], -tie_rank[index]),
            )
            credits[selected_source] -= total_weight
            occurrence_index = source_counts[selected_source]
            source_counts[selected_source] = occurrence_index + 1
            if position >= absolute_start:
                order = source_orders[selected_source]
                selected.append(order[occurrence_index % len(order)])

        order_payload = f"prompt-epoch-order\0{int(seed)}\0{epoch}".encode("utf-8")
        order_seed = int.from_bytes(
            hashlib.sha256(order_payload).digest()[:8],
            "little",
        )
        random.Random(order_seed).shuffle(selected)
        return selected

    def sample_for_epoch(
        self,
        count: int,
        seed: int,
        epoch: int,
        mode: str = "independent_epoch",
        source_weights: Sequence[float] | None = None,
    ) -> list[PromptRecord]:
        """Sample candidates using the legacy or per-source no-repeat schedule."""

        mode = str(mode)
        if mode == "independent_epoch":
            if source_weights is not None:
                raise ValueError(
                    "candidate prompt source weights require no_repeat_cycle mode"
                )
            return self.sample(count, seed=int(seed) + int(epoch))
        if mode == "no_repeat_cycle":
            return self.sample_no_repeat_cycle(
                count,
                seed=seed,
                epoch=epoch,
                source_weights=source_weights,
            )
        raise ValueError(
            "candidate prompt sampling mode must be independent_epoch or "
            f"no_repeat_cycle, got {mode!r}"
        )

    def sample(
        self,
        count: int,
        seed: int,
        excluded_prompts: Iterable[str] = (),
    ) -> list[PromptRecord]:
        """Draw a balanced sample without replacement within each source."""

        count = int(count)
        if count < 1:
            raise ValueError(f"prompt sample count must be positive, got {count}")

        excluded = set(excluded_prompts)
        generator = random.Random(int(seed))
        source_order = list(range(len(self._sources)))
        generator.shuffle(source_order)
        quotas = [count // len(self._sources)] * len(self._sources)
        for source_index in source_order[: count % len(self._sources)]:
            quotas[source_index] += 1

        selected = []
        for source_index, (source, quota) in enumerate(zip(self._sources, quotas, strict=True)):
            available = [record for record in source if record.text not in excluded]
            if quota > len(available):
                raise ValueError(
                    f"prompt source {self.paths[source_index]} has {len(available)} eligible rows, "
                    f"but its balanced quota is {quota}"
                )
            selected.extend(generator.sample(available, quota))

        generator.shuffle(selected)
        return selected


def same_prompt_reference_seed(
    base_seed: int,
    epoch: int,
    prompt: str,
    reference_index: int,
) -> int:
    """Return a stable independent seed for one same-prompt reference."""

    payload = (
        f"{int(base_seed)}\0{int(epoch)}\0"
        f"{int(reference_index)}\0{str(prompt)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


# same as get_loguniform_schedule here https://github.com/ohayonguy/information-estimation-metric/blob/main/information_estimation_metric.py
def iem_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_steps: int,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return descending log-uniform VE sigma boundaries."""

    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    num_steps = int(num_steps)
    if not (math.isfinite(sigma_min) and math.isfinite(sigma_max)):
        raise ValueError("IEM sigma bounds must be finite")
    if not 0 < sigma_min < sigma_max:
        raise ValueError(f"expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}")
    if num_steps < 1:
        raise ValueError(f"IEM needs at least one integration interval, got {num_steps}")
    log_sigma = torch.linspace(
        math.log(sigma_max),
        math.log(sigma_min),
        num_steps + 1,
        device=device,
        dtype=torch.float32,
    )
    return log_sigma.exp()

# same as get_brownian_motion here https://github.com/ohayonguy/information-estimation-metric/blob/main/information_estimation_metric.py
def iem_schedule_terms(
    sigma_schedule: Tensor | Sequence[float],
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return probe sigmas and ascending ``delta_gamma`` integration widths."""

    schedule = torch.as_tensor(sigma_schedule, device=device, dtype=torch.float32)
    if schedule.ndim != 1 or schedule.numel() < 2:
        raise ValueError("sigma_schedule must be one-dimensional with at least two boundaries")
    if not torch.isfinite(schedule).all() or not (schedule > 0).all():
        raise ValueError("sigma_schedule must contain finite positive values")
    gamma_boundaries = schedule.reciprocal().square()
    delta_gamma = gamma_boundaries[1:] - gamma_boundaries[:-1]
    if not (delta_gamma > 0).all():
        raise ValueError("sigma_schedule must descend strictly so gamma ascends")
    return schedule[:-1], delta_gamma


def sample_iem_noise_table(
    sigma_schedule: Tensor | Sequence[float],
    signal_shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    seed: int,
) -> Tensor:
    """Draw one IID Gaussian tensor per equation-16 integration level."""

    probe_sigmas, _ = iem_schedule_terms(sigma_schedule, device=device)
    shape = tuple(int(size) for size in signal_shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"signal_shape must be positive, got {shape}")
    generator = torch.Generator(device=probe_sigmas.device).manual_seed(int(seed))
    return torch.randn(
        (probe_sigmas.numel(), 1, *shape),
        device=probe_sigmas.device,
        dtype=dtype,
        generator=generator,
    )


def iem_noise_table_seed(seed: int, epoch: int, table_count: int, table_index: int) -> int:
    """Return the deterministic seed for one epoch's IEM noise table."""
    table_count = int(table_count)
    table_index = int(table_index)
    if table_count < 1 or not 0 <= table_index < table_count:
        raise ValueError("IEM noise table index is outside the configured range")
    return (
        int(seed)
        + IEM_NOISE_SEED_OFFSET
        + int(epoch) * table_count
        + table_index
    )


def iem_noise_sha256(noise_table: Tensor) -> str:
    """Hash the exact FP32 IEM noise values used by candidates and references."""
    noise = torch.as_tensor(noise_table).detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(noise.numpy().tobytes(order="C")).hexdigest()


def repeat_endpoint_conditions(
    prompt_embeds: Tensor,
    pooled_prompt_embeds: Tensor,
    flattened_batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Repeat endpoint conditions in the level-major order used by IEM probes."""

    if prompt_embeds.shape[0] < 1 or prompt_embeds.shape[0] != pooled_prompt_embeds.shape[0]:
        raise ValueError("endpoint prompt-condition batches must be equal and nonempty")
    flattened_batch_size = int(flattened_batch_size)
    endpoint_count = prompt_embeds.shape[0]
    if flattened_batch_size < endpoint_count or flattened_batch_size % endpoint_count:
        raise ValueError("flattened probe count must be a positive multiple of endpoint count")
    repeats = flattened_batch_size // endpoint_count
    return (
        prompt_embeds.repeat((repeats,) + (1,) * (prompt_embeds.ndim - 1)),
        pooled_prompt_embeds.repeat((repeats,) + (1,) * (pooled_prompt_embeds.ndim - 1)),
    )


@torch.no_grad()
def iem_features(
    x_0: Tensor,
    velocity_prediction: Callable[[Tensor, Tensor], Tensor],
    sigma_schedule: Tensor | Sequence[float],
    noise_table: Tensor,
    level_batch_size: int = 4,
) -> Tensor:
    """Compute equation 16 with independent noise levels in batched forwards.

    SD3 predicts rectified-flow velocity ``v = epsilon - x_0`` at
    ``x_t = (1-t)x_0 + t epsilon``. Consequently the clean prediction requested
    by equation 14 is ``f(x_t) = x_t - t v(x_t, t)``.

    Each denoiser call handles at most ``level_batch_size`` integration levels.
    Inputs are flattened in level-major order, and every level's flow time is
    repeated once per endpoint so sigma, noise, timestep, and condition stay
    aligned. Setting ``level_batch_size`` to the number of levels performs one
    denoiser forward; smaller values bound activation memory without changing
    the result.
    """

    x_0 = torch.as_tensor(x_0).float()
    if x_0.ndim < 2 or x_0.shape[0] < 1:
        raise ValueError(f"x_0 must have nonempty (batch, ...) shape, got {tuple(x_0.shape)}")
    probe_sigmas, delta_gamma = iem_schedule_terms(sigma_schedule, device=x_0.device)
    interval_count = probe_sigmas.numel()
    level_batch_size = int(level_batch_size)
    if level_batch_size < 1:
        raise ValueError(f"level_batch_size must be positive, got {level_batch_size}")
    if interval_count % level_batch_size:
        raise ValueError(
            f"level_batch_size ({level_batch_size}) must divide the number "
            f"of IEM levels ({interval_count}) exactly"
        )

    noise_table = torch.as_tensor(noise_table)
    expected_tail = tuple(x_0.shape[1:])
    if (
        noise_table.ndim != x_0.ndim + 1
        or noise_table.shape[0] != interval_count
        or noise_table.shape[1] not in (1, x_0.shape[0])
        or tuple(noise_table.shape[2:]) != expected_tail
    ):
        raise ValueError(
            "noise_table must have shape (levels, 1|batch, *signal_shape); "
            f"got {tuple(noise_table.shape)}"
        )

    feature_blocks = []
    broadcast_tail = (1,) * (x_0.ndim - 1)
    for start in range(0, interval_count, level_batch_size):
        stop = min(start + level_batch_size, interval_count)
        sigma = probe_sigmas[start:stop]
        flow_time = sigma / (1.0 + sigma)
        noise = noise_table[start:stop].to(device=x_0.device, dtype=x_0.dtype)
        x_t = (
            x_0.unsqueeze(0) + sigma.reshape((-1, 1) + broadcast_tail) * noise
        ) / (1.0 + sigma).reshape((-1, 1) + broadcast_tail)

        # x_t is [level, endpoint, ...], so flattening is level-major. Match it
        # with [t_0 repeated B times, t_1 repeated B times, ...].
        flat_x_t = x_t.flatten(0, 1)
        flat_t = flow_time.repeat_interleave(x_0.shape[0])
        velocity = velocity_prediction(flat_x_t, flat_t).float().reshape_as(x_t)
        f_x_t = x_t - flow_time.reshape((-1, 1) + broadcast_tail) * velocity
        e_gamma = x_0.unsqueeze(0) - f_x_t
        weights = delta_gamma[start:stop].sqrt()
        weighted = weights.reshape((-1, 1) + broadcast_tail) * e_gamma
        feature_blocks.append(weighted.transpose(0, 1).flatten(start_dim=1))

    return torch.cat(feature_blocks, dim=1)


def update_reference_sums(
    count: int,
    sum_phi: Tensor | None,
    sum_phi_squared_norm: Tensor | None,
    features: Tensor,
) -> tuple[int, Tensor, Tensor]:
    """Accumulate ``M``, ``sum Phi(x'_j)``, and ``sum ||Phi(x'_j)||^2``."""

    features = torch.as_tensor(features).float()
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")
    features_64 = features.to(dtype=torch.float64)
    batch_sum_phi = features_64.sum(dim=0)
    batch_sum_phi_squared_norm = features_64.square().sum()

    count = int(count)
    if count == 0:
        if sum_phi is not None or sum_phi_squared_norm is not None:
            raise ValueError("zero reference count must have empty sums")
        return features.shape[0], batch_sum_phi, batch_sum_phi_squared_norm
    if count < 0 or sum_phi is None or sum_phi_squared_norm is None:
        raise ValueError("positive reference count requires both accumulated sums")

    sum_phi = torch.as_tensor(sum_phi, device=features.device, dtype=torch.float64)
    if sum_phi.shape != features.shape[1:]:
        raise ValueError("accumulated feature sum has the wrong shape")
    sum_phi_squared_norm = torch.as_tensor(
        sum_phi_squared_norm,
        device=features.device,
        dtype=torch.float64,
    )
    if sum_phi_squared_norm.numel() != 1:
        raise ValueError("accumulated squared-norm sum must be scalar")
    return (
        count + features.shape[0],
        sum_phi + batch_sum_phi,
        sum_phi_squared_norm + batch_sum_phi_squared_norm,
    )


def finalize_reference_statistics(
    count: int, sum_phi: Tensor, sum_phi_squared_norm: Tensor
) -> tuple[Tensor, Tensor]:
    """Return equation-21 ``mu_omega`` and ``v_omega`` from accumulated sums."""
    count = int(count)
    if count < 1:
        raise ValueError("reference statistics require a positive count")
    sum_phi = torch.as_tensor(sum_phi, dtype=torch.float64)
    sum_phi_squared_norm = torch.as_tensor(sum_phi_squared_norm, dtype=torch.float64)
    if sum_phi.ndim != 1 or sum_phi_squared_norm.numel() != 1:
        raise ValueError("reference sums have incompatible shapes")
    mu_omega_64 = sum_phi / count
    v_omega = (sum_phi_squared_norm / count - mu_omega_64.square().sum()).clamp_min(0)
    return mu_omega_64.float(), v_omega


def iem_reward_from_reference_statistics(
    features: Tensor,
    mu_omega: Tensor,
    v_omega: Tensor | float,
) -> Tensor:
    """Evaluate ``||Phi(x) - mu_omega||^2 + v_omega`` from equation 21."""

    features = torch.as_tensor(features).float()
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")
    mu_omega = torch.as_tensor(mu_omega, device=features.device, dtype=torch.float32)
    if mu_omega.shape != features.shape[1:]:
        raise ValueError("mu_omega and candidate feature dimensions do not match")
    v_omega = torch.as_tensor(v_omega, device=features.device, dtype=torch.float64)
    if v_omega.numel() != 1 or not torch.isfinite(v_omega) or v_omega < 0:
        raise ValueError("v_omega must be a finite non-negative scalar")
    return (features - mu_omega).square().sum(dim=1, dtype=torch.float64) + v_omega


def negative_g_reward_from_reference_mean(
    features: Tensor,
    mu_omega: Tensor,
) -> Tensor:
    """Evaluate equation 10's ``-g(x) = -Phi(x)^T mu_omega``.

    Equation 16 already includes the quadrature weights and shared noise table.
    Averaging ``Phi(x)^T Phi(X)`` over references therefore only requires
    their feature mean ``mu_omega``.
    """

    features = torch.as_tensor(features)
    if features.ndim != 2:
        raise ValueError(
            f"features must have shape (N, D), got {tuple(features.shape)}"
        )
    mu_omega = torch.as_tensor(mu_omega, device=features.device)
    if mu_omega.shape != features.shape[1:]:
        raise ValueError("mu_omega and candidate feature dimensions do not match")
    return -(features.double() * mu_omega.double()).sum(dim=1)


def log_density_from_iem_features(
    features: Tensor,
    sigma_schedule: Tensor | Sequence[float],
    signal_dimension: int,
    *,
    feature_chunk_size: int = 65_536,
) -> Tensor:
    """Estimate conditional ``log p(x)`` from equation-16 features.

    The finite density identity is

    ``log p(x) = 1/2 sum_i [d/(1+gamma_i) - ||e_i(x)||^2]``
    ``             * delta_gamma_i - d/2 log(2*pi*e)``.

    Every equation-16 feature block is ``sqrt(delta_gamma_i) * e_i(x)``.
    Therefore ``||Phi(x)||^2`` is already the unaveraged residual sum; no
    division or multiplication by the number of integration levels belongs
    here. The denoiser condition and noise table are exactly those used to
    construct ``features``.
    """

    features = torch.as_tensor(features)
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(
            "features must have nonempty shape (batch, feature_dimension)"
        )
    signal_dimension = int(signal_dimension)
    if signal_dimension < 1:
        raise ValueError("signal_dimension must be positive")
    feature_chunk_size = int(feature_chunk_size)
    if feature_chunk_size < 1:
        raise ValueError("feature_chunk_size must be positive")

    probe_sigmas, delta_gamma = iem_schedule_terms(
        sigma_schedule,
        device=features.device,
    )
    expected_feature_dimension = signal_dimension * probe_sigmas.numel()
    if features.shape[1] != expected_feature_dimension:
        raise ValueError(
            "feature dimension must equal signal_dimension times the number "
            f"of IEM levels; got {features.shape[1]} and expected "
            f"{expected_feature_dimension}"
        )

    gamma = probe_sigmas.double().reciprocal().square()
    normal_integral = (
        signal_dimension / (1.0 + gamma) * delta_gamma.double()
    ).sum()
    feature_norm_squared = torch.zeros(
        features.shape[0],
        device=features.device,
        dtype=torch.float64,
    )
    for start in range(0, features.shape[1], feature_chunk_size):
        stop = min(start + feature_chunk_size, features.shape[1])
        feature_block = features[:, start:stop].double()
        feature_norm_squared += feature_block.square().sum(dim=1)

    entropy_constant = 0.5 * signal_dimension * math.log(
        2.0 * math.pi * math.e
    )
    return 0.5 * (normal_integral - feature_norm_squared) - entropy_constant


def negative_log_p_plus_negative_g_reward(
    features: Tensor,
    mu_omega: Tensor,
    sigma_schedule: Tensor | Sequence[float],
    signal_dimension: int,
    *,
    a1: float = 1.0,
    a2: float = 1.0,
) -> Tensor:
    """Return ``-a1 * log p(x) + a2 * (-g(x))``.

    ``-g(x)`` is equation 10 evaluated against the empirical reference mean.
    Both terms reuse the same candidate feature vector, baseline denoiser,
    prompt condition, integration schedule, and noise table.
    """

    a1 = float(a1)
    a2 = float(a2)
    if not math.isfinite(a1) or a1 < 0.0:
        raise ValueError("a1 must be finite and non-negative")
    if not math.isfinite(a2) or a2 < 0.0:
        raise ValueError("a2 must be finite and non-negative")
    if a1 == 0.0 and a2 == 0.0:
        raise ValueError("a1 and a2 cannot both be zero")

    log_p = log_density_from_iem_features(
        features,
        sigma_schedule,
        signal_dimension,
    )
    negative_g = negative_g_reward_from_reference_mean(features, mu_omega)
    return -a1 * log_p + a2 * negative_g


def iem_objective_from_reference_statistics(
    features: Tensor,
    mu_omega: Tensor,
    v_omega: Tensor | float,
    objective: str,
    *,
    sigma_schedule: Tensor | Sequence[float] | None = None,
    signal_dimension: int | None = None,
    density_a1: float = 1.0,
    density_a2: float = 1.0,
) -> Tensor:
    """Evaluate the configured objective from one shared IEM reference cloud."""

    objective = str(objective).lower()
    if objective == "expected_squared_distance":
        return iem_reward_from_reference_statistics(features, mu_omega, v_omega)
    if objective == "negative_g":
        return negative_g_reward_from_reference_mean(features, mu_omega)
    if objective == "negative_log_p_plus_negative_g":
        if sigma_schedule is None or signal_dimension is None:
            raise ValueError(
                "negative_log_p_plus_negative_g requires sigma_schedule "
                "and signal_dimension"
            )
        return negative_log_p_plus_negative_g_reward(
            features,
            mu_omega,
            sigma_schedule,
            signal_dimension,
            a1=density_a1,
            a2=density_a2,
        )
    raise ValueError("IEM objective must be one of: " + ", ".join(IEM_OBJECTIVES))


def update_reference_feature_sum(
    count: int,
    sum_features: Tensor | None,
    features: Tensor,
) -> tuple[int, Tensor]:
    """Accumulate image embeddings without extra moments."""

    features = torch.as_tensor(features).float()
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError(
            f"features must have shape (N, D), got {tuple(features.shape)}"
        )
    batch_sum = features.to(dtype=torch.float64).sum(dim=0)
    count = int(count)
    if count == 0:
        if sum_features is not None:
            raise ValueError("zero reference count must have an empty feature sum")
        return features.shape[0], batch_sum
    if count < 0 or sum_features is None:
        raise ValueError("positive reference count requires an accumulated feature sum")
    sum_features = torch.as_tensor(
        sum_features,
        device=features.device,
        dtype=torch.float64,
    )
    if sum_features.shape != features.shape[1:]:
        raise ValueError("accumulated feature sum has the wrong shape")
    return count + features.shape[0], sum_features + batch_sum


def cosine_distance_from_reference_mean(
    candidate_features: Tensor,
    mean_reference_feature: Tensor,
) -> Tensor:
    """Return the mean cosine distance from each candidate to a reference group.

    Reference and candidate rows are normalized before their statistics are
    formed. The arithmetic reference mean must not itself be normalized because
    1 - c @ mean(r_j) is exactly mean_j(1 - cosine(c, r_j)).
    """

    candidates = torch.as_tensor(candidate_features).float()
    reference_mean = torch.as_tensor(
        mean_reference_feature,
        device=candidates.device,
        dtype=torch.float32,
    )
    if candidates.ndim != 2 or reference_mean.shape != candidates.shape[1:]:
        raise ValueError("candidate features and reference mean are incompatible")
    candidates = F.normalize(candidates, p=2, dim=1)
    return (1.0 - candidates @ reference_mean).clamp(0.0, 2.0)


def mean_nearest_cosine_distance(
    candidate_features: Tensor,
    reference_features: Tensor,
    fraction: float,
) -> tuple[Tensor, int]:
    """Average each candidate's nearest fraction of reference distances."""

    candidates = torch.as_tensor(candidate_features).float()
    references = torch.as_tensor(
        reference_features,
        device=candidates.device,
        dtype=torch.float32,
    )
    if (
        candidates.ndim != 2
        or references.ndim != 2
        or candidates.shape[1] != references.shape[1]
    ):
        raise ValueError(
            "candidate and reference features must be compatible matrices"
        )
    if candidates.shape[0] < 1 or references.shape[0] < 1:
        raise ValueError("candidate and reference feature matrices must be nonempty")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-reference fraction must be in (0, 1]")

    candidates = F.normalize(candidates, p=2, dim=1)
    references = F.normalize(references, p=2, dim=1)
    distances = (1.0 - candidates @ references.T).clamp(0.0, 2.0)
    selected_count = max(1, math.ceil(fraction * references.shape[0]))
    selected_distances = torch.topk(
        distances,
        selected_count,
        dim=1,
        largest=False,
        sorted=False,
    ).values
    return selected_distances.mean(dim=1), selected_count


def pairwise_l2_distances(
    candidate_features: Tensor,
    reference_features: Tensor,
) -> Tensor:
    """Return raw-embedding L2 distances for every candidate/reference pair."""

    candidates = torch.as_tensor(candidate_features).float()
    references = torch.as_tensor(
        reference_features,
        device=candidates.device,
        dtype=torch.float32,
    )
    if (
        candidates.ndim != 2
        or references.ndim != 2
        or candidates.shape[1] != references.shape[1]
    ):
        raise ValueError(
            "candidate and reference features must be compatible matrices"
        )
    if candidates.shape[0] < 1 or references.shape[0] < 1:
        raise ValueError(
            "candidate and reference feature matrices must be nonempty"
        )
    return torch.cdist(candidates, references, p=2)


def mean_pairwise_l2_distance(
    candidate_features: Tensor,
    reference_features: Tensor,
) -> Tensor:
    """Average each candidate's raw-embedding L2 distance over references."""

    return pairwise_l2_distances(
        candidate_features,
        reference_features,
    ).mean(dim=1)


def mean_nearest_l2_distance(
    candidate_features: Tensor,
    reference_features: Tensor,
    fraction: float,
) -> tuple[Tensor, int]:
    """Average each candidate's nearest raw-embedding L2 distances."""

    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("nearest-reference fraction must be in (0, 1]")
    distances = pairwise_l2_distances(candidate_features, reference_features)
    selected_count = max(1, math.ceil(fraction * distances.shape[1]))
    selected_distances = torch.topk(
        distances,
        selected_count,
        dim=1,
        largest=False,
        sorted=False,
    ).values
    return selected_distances.mean(dim=1), selected_count


class ReferenceReservoir:
    """Bounded priority reservoir of reference-model latents and prompt IDs."""

    latent_dtype = torch.bfloat16

    def __init__(self, capacity: int, prompt_source_sha256: str):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError(f"reference reservoir capacity must be positive, got {capacity}")
        self.prompt_source_sha256 = str(prompt_source_sha256)
        self.latents: Tensor | None = None
        self.prompt_ids: Tensor | None = None
        self.priorities: Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.latents is None else self.latents.shape[0]

    def clear(self) -> None:
        """Discard endpoints produced by an obsolete reference snapshot."""

        self.latents = self.prompt_ids = self.priorities = None

    def update(self, latents: Tensor, prompt_ids: Tensor, seed: int) -> None:
        latents = torch.as_tensor(latents).detach().to(device="cpu", dtype=self.latent_dtype)
        prompt_ids = torch.as_tensor(prompt_ids).detach().to(device="cpu", dtype=torch.int64)
        if latents.ndim < 2 or prompt_ids.ndim != 1 or latents.shape[0] != prompt_ids.shape[0]:
            raise ValueError("reference latents and prompt ids must be row-aligned and nonempty")
        if latents.shape[0] < 1:
            raise ValueError("cannot update the reference reservoir with an empty batch")
        if self.latents is not None and tuple(self.latents.shape[1:]) != tuple(latents.shape[1:]):
            raise ValueError("new reference latent shape does not match the reservoir")

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        priorities = torch.rand(latents.shape[0], generator=generator, dtype=torch.float32)
        if self.latents is not None:
            latents = torch.cat((self.latents, latents))
            prompt_ids = torch.cat((self.prompt_ids, prompt_ids))
            priorities = torch.cat((self.priorities, priorities))
        if latents.shape[0] > self.capacity:
            selected = torch.topk(priorities, self.capacity, sorted=False).indices
            latents = latents[selected]
            prompt_ids = prompt_ids[selected]
            priorities = priorities[selected]
        self.latents = latents.contiguous()
        self.prompt_ids = prompt_ids.contiguous()
        self.priorities = priorities.contiguous()

    def subset(self, count: int, seed: int) -> tuple[Tensor, Tensor]:
        if self.latents is None:
            raise RuntimeError("the reference reservoir is empty")
        count = min(int(count), len(self))
        if count < 1:
            raise ValueError(f"reference subset size must be positive, got {count}")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        indices = torch.randperm(len(self), generator=generator)[:count]
        return self.latents[indices], self.prompt_ids[indices]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "prompt_source_sha256": self.prompt_source_sha256,
            "latents": self.latents,
            "prompt_ids": self.prompt_ids,
            "priorities": self.priorities,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("checkpoint reference-reservoir capacity does not match the config")
        if state["prompt_source_sha256"] != self.prompt_source_sha256:
            raise ValueError("checkpoint reference prompt sources do not match the configured files")
        values = (state.get("latents"), state.get("prompt_ids"), state.get("priorities"))
        if all(value is None for value in values):
            self.latents = self.prompt_ids = self.priorities = None
            return
        if any(value is None for value in values):
            raise ValueError("checkpoint reference-reservoir state is incomplete")
        self.latents = torch.as_tensor(values[0], device="cpu", dtype=self.latent_dtype)
        self.prompt_ids = torch.as_tensor(values[1], device="cpu", dtype=torch.int64)
        self.priorities = torch.as_tensor(values[2], device="cpu", dtype=torch.float32)
        if (
            self.prompt_ids.ndim != 1
            or self.priorities.ndim != 1
            or self.latents.shape[0] != self.prompt_ids.shape[0]
            or self.latents.shape[0] != self.priorities.shape[0]
            or len(self) > self.capacity
        ):
            raise ValueError("checkpoint reference-reservoir tensors are not aligned")


class IEMReward:
    """Stateful reference-distance reward backed by a frozen reference model."""

    def __init__(
        self,
        pipe,
        model,
        accelerator,
        config,
        reference_latent_generator: Callable[..., Tensor] | None = None,
        reference_adapter_name: str | None = None,
    ):
        self.pipe = pipe
        self.model = model
        self.accelerator = accelerator
        self.config = config
        self.reference_adapter_name = reference_adapter_name
        self.reference_model_update = ReferenceModelUpdatePolicy.from_config(
            config
        )
        self.reference_latent_generator = reference_latent_generator
        self.distance_metric = str(
            getattr(config, "distance_metric", "iem")
        ).lower()
        self.iem_objective = str(
            getattr(config, "iem_objective", "expected_squared_distance")
        ).lower()
        self.density_a1 = float(getattr(config, "density_a1", 1.0))
        self.density_a2 = float(getattr(config, "density_a2", 1.0))
        self.reference_prompt_mode = str(
            getattr(config, "reference_prompt_mode", "diverse")
        )
        self.reference_selection_mode = str(
            getattr(config, "reference_selection_mode", "all")
        ).lower()
        self.nearest_reference_fraction = float(
            getattr(config, "nearest_reference_fraction", 0.1)
        )
        self._image_processor = None
        self._image_encoder = None
        self.reference_cache_use_iem_statistics = bool(
            getattr(config, "reference_cache_use_iem_statistics", True)
        )
        reference_cache_dir = getattr(config, "reference_cache_dir", None)
        self.reference_cache = (
            ReferenceCacheReader(
                reference_cache_dir,
                expected_spec_sha256=getattr(
                    config,
                    "reference_cache_spec_sha256",
                    None,
                ),
            )
            if reference_cache_dir
            else None
        )
        self.sigma_schedule = (
            iem_sigma_schedule(
                config.sigma_min,
                config.sigma_max,
                config.num_steps,
                device=accelerator.device,
            )
            if self.distance_metric == "iem"
            else None
        )
        self._validate_config()
        if self.reference_cache is not None:
            self._validate_reference_cache()
        if self.reference_prompt_mode == "diverse":
            self.prompt_sampler = BalancedPromptSampler(
                config.reference_prompt_files
            )
            self.reservoir = ReferenceReservoir(
                config.reference_bank_size,
                self.prompt_sampler.source_sha256,
            )
        else:
            self.prompt_sampler = None
            self.reservoir = None

    def _reference_pipe(self, *args, **kwargs):
        """Generate baseline endpoints through the host sampler when supplied."""
        if self.reference_latent_generator is not None:
            kwargs.pop("output_type", None)
            kwargs.pop("return_dict", None)
            return (self.reference_latent_generator(*args, **kwargs),)
        return self.pipe(*args, **kwargs)

    @property
    def reward_log_name(self) -> str:
        if self.distance_metric == "iem" and self.iem_objective == "negative_g":
            return "NegativeG"
        if (
            self.distance_metric == "iem"
            and self.iem_objective == "negative_log_p_plus_negative_g"
        ):
            return "NegativeLogPPlusNegativeG"
        return {
            "iem": "IEM",
            "clip_cosine": "CLIPCosineDistance",
            "clip_l2": "CLIPL2Distance",
            "dino_cosine": "DINOCosineDistance",
            "dino_l2": "DINOL2Distance",
            "tpips_overall": "TPIPSOverallDistance",
        }[self.distance_metric]

    def _validate_config(self) -> None:
        if self.distance_metric not in SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                "creativity.distance_metric must be one of: "
                + ", ".join(SUPPORTED_DISTANCE_METRICS)
            )
        if self.reference_prompt_mode not in ("diverse", "same_prompt"):
            raise ValueError(
                "creativity.reference_prompt_mode must be 'diverse' "
                "or 'same_prompt'"
            )
        if self.reference_selection_mode not in ("all", "nearest"):
            raise ValueError(
                "creativity.reference_selection_mode must be 'all' or 'nearest'"
            )
        if self.iem_objective not in IEM_OBJECTIVES:
            raise ValueError(
                "creativity.iem_objective must be one of: "
                + ", ".join(IEM_OBJECTIVES)
            )
        if (
            self.distance_metric == "iem"
            and self.iem_objective in REFERENCE_MEAN_IEM_OBJECTIVES
            and self.reference_selection_mode != "all"
        ):
            raise ValueError(
                f"creativity.iem_objective={self.iem_objective!r} requires "
                "reference_selection_mode='all'"
            )
        if (
            self.distance_metric == "iem"
            and self.iem_objective == "negative_log_p_plus_negative_g"
        ):
            if not math.isfinite(self.density_a1) or self.density_a1 < 0.0:
                raise ValueError(
                    "creativity.density_a1 must be finite and non-negative"
                )
            if not math.isfinite(self.density_a2) or self.density_a2 < 0.0:
                raise ValueError(
                    "creativity.density_a2 must be finite and non-negative"
                )
            if self.density_a1 == 0.0 and self.density_a2 == 0.0:
                raise ValueError(
                    "creativity.density_a1 and density_a2 cannot both be zero"
                )
        if not 0.0 < self.nearest_reference_fraction <= 1.0:
            raise ValueError(
                "creativity.nearest_reference_fraction must be in (0, 1]"
            )
        if (
            self.distance_metric == "iem"
            and self.reference_selection_mode == "nearest"
        ):
            raise ValueError(
                "nearest reference selection is not yet supported for "
                "creativity.distance_metric='iem'"
            )
        positive_fields = [
            "reference_batch_size",
            "feature_batch_size",
        ]
        if self.distance_metric == "iem":
            positive_fields.extend(("level_batch_size", "noise_table_count"))
        elif self.distance_metric == "tpips_overall":
            positive_fields.append("tpips_batch_size")
        if self.reference_prompt_mode == "diverse":
            positive_fields.extend(
                (
                    "reference_samples_per_epoch",
                    "reference_subset_size",
                    "reference_bank_size",
                )
            )
        else:
            positive_fields.append("reference_samples_per_prompt")
        for name in positive_fields:
            if int(getattr(self.config, name)) < 1:
                raise ValueError(f"creativity.{name} must be positive")
        if (
            self.reference_prompt_mode == "diverse"
            and int(self.config.reference_samples_per_epoch)
            % self.accelerator.num_processes
        ):
            raise ValueError("creativity.reference_samples_per_epoch must be divisible by the process count")
        if self.distance_metric != "iem":
            model_field = IMAGE_DISTANCE_MODEL_FIELDS[self.distance_metric]
            if not str(getattr(self.config, model_field, "")):
                raise ValueError(f"creativity.{model_field} must be nonempty")
        if (
            self.reference_model_update.adapter_name
            != self.reference_adapter_name
        ):
            raise ValueError(
                "the rolling reference-model adapter does not match "
                "creativity.reference_model_update.enabled"
            )


    def _validate_reference_cache(self) -> None:
        """Reject cached references generated with incompatible RAM inputs."""

        if self.reference_prompt_mode != "same_prompt":
            raise ValueError(
                "creativity.reference_cache_dir is supported only with "
                "reference_prompt_mode=same_prompt"
            )
        if self.distance_metric == "tpips_overall":
            raise ValueError(
                "the RAM reference cache contains IEM, CLIP, and DINO data, "
                "not TPIPS embeddings"
            )
        candidate_prompt_files = getattr(
            self.config,
            "candidate_prompt_files",
            None,
        )
        if not candidate_prompt_files:
            raise ValueError(
                "creativity.candidate_prompt_files is required when using "
                "the RAM reference cache"
            )
        spec = self.reference_cache.spec
        expected = {
            "seed": int(self.config.seed),
            "resolution": int(self.config.resolution),
            "num_inference_steps": int(self.config.num_inference_steps),
            "guidance_scale": float(self.config.guidance_scale),
            "reference_samples_per_prompt": int(
                self.config.reference_samples_per_prompt
            ),
            "prompt_source_sha256": BalancedPromptSampler(
                candidate_prompt_files
            ).source_sha256,
        }
        mismatches = {
            name: (spec.get(name), value)
            for name, value in expected.items()
            if spec.get(name) != value
        }
        for name, model_id in {
            "clip": str(self.config.clip_model_id),
            "dino": str(self.config.dino_model_id),
        }.items():
            cached = spec.get(name)
            if not isinstance(cached, dict) or cached.get("model_id") != model_id:
                mismatches[f"{name}.model_id"] = (
                    cached.get("model_id") if isinstance(cached, dict) else None,
                    model_id,
                )
        if (
            self.distance_metric == "iem"
            and self.reference_cache_use_iem_statistics
        ):
            cached_iem = spec.get("iem")
            expected_iem = {
                "sigma_min": float(self.config.sigma_min),
                "sigma_max": float(self.config.sigma_max),
                "num_steps": int(self.config.num_steps),
                "noise_table_count": int(self.config.noise_table_count),
                "world_size": int(self.accelerator.num_processes),
                "feature_weighting": IEM_FEATURE_WEIGHTING,
                "noise_assignment": IEM_NOISE_ASSIGNMENT,
            }
            if not isinstance(cached_iem, dict):
                mismatches["iem"] = (cached_iem, expected_iem)
            else:
                for name, value in expected_iem.items():
                    if cached_iem.get(name) != value:
                        mismatches[f"iem.{name}"] = (
                            cached_iem.get(name),
                            value,
                        )

        if mismatches:
            raise ValueError(
                "RAM reference cache is incompatible with NFT creativity "
                f"config: {mismatches}"
            )

    def _reference_cache_active(self, epoch: int) -> bool:
        update_policy = getattr(
            self, "reference_model_update", ReferenceModelUpdatePolicy()
        )
        return (
            self.reference_cache is not None
            and update_policy.cache_allowed(epoch)
        )

    def _cached_references(
        self,
        *,
        epoch: int,
        prompt: str,
        tensor_keys: Sequence[str],
    ) -> dict[str, Tensor] | None:
        if not self._reference_cache_active(epoch):
            return None
        if not self.reference_cache.has_entry(epoch=epoch, prompt=prompt):
            return None
        tensors = self.reference_cache.load(
            epoch=epoch,
            prompt=prompt,
            tensor_keys=tensor_keys,
        )
        expected_count = int(self.config.reference_samples_per_prompt)
        if any(tensor.shape[0] != expected_count for tensor in tensors.values()):
            raise ValueError(
                "RAM reference cache entry count does not match "
                "reference_samples_per_prompt"
            )
        return tensors

    def _cached_iem_statistics(
        self,
        *,
        epoch: int,
        prompt: str,
        noise_seed: int,
        noise_sha256: str,
        feature_dimension: int,
    ) -> tuple[Tensor, Tensor] | None:
        """Load statistics only when they match the candidate's exact noise."""
        if (
            not self._reference_cache_active(epoch)
            or self.reference_selection_mode != "all"
            or not self.reference_cache_use_iem_statistics
        ):
            return None
        if not self.reference_cache.has_entry(epoch=epoch, prompt=prompt):
            return None
        tensors = self.reference_cache.load(
            epoch=epoch,
            prompt=prompt,
            tensor_keys=(
                IEM_MEAN_KEY,
                IEM_VARIANCE_KEY,
                IEM_NOISE_SEED_KEY,
                IEM_NOISE_SHA256_KEY,
            ),
        )
        saved_seed = int(tensors[IEM_NOISE_SEED_KEY].item())
        if saved_seed != int(noise_seed):
            raise ValueError("cached IEM noise seed does not match candidate noise")
        saved_sha256 = bytes(tensors[IEM_NOISE_SHA256_KEY].tolist()).hex()
        if saved_sha256 != str(noise_sha256):
            raise ValueError("cached IEM noise hash does not match candidate noise")
        mean = tensors[IEM_MEAN_KEY]
        variance = tensors[IEM_VARIANCE_KEY]
        if mean.ndim != 1 or mean.numel() != int(feature_dimension):
            raise ValueError("cached IEM mean has the wrong feature dimension")
        if variance.numel() != 1 or not torch.isfinite(variance).all():
            raise ValueError("cached IEM variance is not a finite scalar")
        return (
            mean.to(self.accelerator.device, dtype=torch.float32),
            variance.to(self.accelerator.device, dtype=torch.float64),
        )

    @staticmethod
    def _adapter_host(model):
        if hasattr(model, "disable_adapter"):
            return model
        if hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
            return model.module
        raise AttributeError("IEM transformer does not expose PEFT adapter controls")

    def _reference_model(self):
        return reference_adapter_context(
            self._adapter_host(self.model),
            self.reference_adapter_name,
        )

    def on_reference_model_updated(self) -> None:
        """Invalidate state containing endpoints from the previous snapshot."""

        if self.reservoir is not None:
            self.reservoir.clear()

    def state_dict(self) -> dict:
        if self.reference_prompt_mode == "diverse":
            state = self.reservoir.state_dict()
        else:
            state = {}
        return {
            "distance_metric": self.distance_metric,
            "iem_objective": self.iem_objective,
            "reference_prompt_mode": self.reference_prompt_mode,
            "reference_selection_mode": self.reference_selection_mode,
            "nearest_reference_fraction": self.nearest_reference_fraction,
            **state,
            **(
                {
                    "density_a1": self.density_a1,
                    "density_a2": self.density_a2,
                }
                if self.distance_metric == "iem"
                and self.iem_objective == "negative_log_p_plus_negative_g"
                else {}
            ),
            **(
                {"reference_cache_spec_sha256": self.reference_cache.spec_sha256}
                if self.reference_cache is not None else {}
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        saved_metric = str(state.get("distance_metric", "iem"))
        saved_iem_objective = str(
            state.get("iem_objective", "expected_squared_distance")
        )
        saved_mode = str(state.get("reference_prompt_mode", "diverse"))
        saved_selection_mode = str(
            state.get("reference_selection_mode", "all")
        )
        if saved_metric != self.distance_metric:
            raise ValueError(
                "checkpoint creativity.distance_metric "
                f"is {saved_metric!r}, but the current config uses "
                f"{self.distance_metric!r}"
            )
        if saved_iem_objective != self.iem_objective:
            raise ValueError(
                "checkpoint creativity.iem_objective "
                f"is {saved_iem_objective!r}, but the current config uses "
                f"{self.iem_objective!r}"
            )
        if (
            self.distance_metric == "iem"
            and self.iem_objective == "negative_log_p_plus_negative_g"
        ):
            for name in ("density_a1", "density_a2"):
                saved_value = float(state.get(name, 1.0))
                configured_value = float(getattr(self, name))
                if not math.isclose(
                    saved_value,
                    configured_value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"checkpoint creativity.{name} is {saved_value}, "
                        f"but the current config uses {configured_value}"
                    )
        if saved_mode != self.reference_prompt_mode:
            raise ValueError(
                "checkpoint creativity.reference_prompt_mode "
                f"is {saved_mode!r}, but the current config uses "
                f"{self.reference_prompt_mode!r}"
            )
        if saved_selection_mode != self.reference_selection_mode:
            raise ValueError(
                "checkpoint creativity.reference_selection_mode "
                f"is {saved_selection_mode!r}, but the current config uses "
                f"{self.reference_selection_mode!r}"
            )
        saved_cache_sha256 = state.get("reference_cache_spec_sha256")
        current_cache_sha256 = (
            self.reference_cache.spec_sha256
            if self.reference_cache is not None else None
        )
        if saved_cache_sha256 != current_cache_sha256:
            raise ValueError(
                "checkpoint creativity.reference_cache_spec_sha256 "
                f"is {saved_cache_sha256!r}, but the current config uses "
                f"{current_cache_sha256!r}"
            )
        if self.reference_selection_mode == "nearest":
            saved_fraction = float(state.get("nearest_reference_fraction", 0.1))
            if not math.isclose(
                saved_fraction,
                self.nearest_reference_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "checkpoint creativity.nearest_reference_fraction "
                    f"is {saved_fraction}, but the current config uses "
                    f"{self.nearest_reference_fraction}"
                )
        if self.reference_prompt_mode == "diverse":
            self.reservoir.load_state_dict(state)

    def _encode_prompts(self, prompts: Sequence[str]) -> tuple[Tensor, Tensor]:
        # Equation-14 probes use positive prompt conditioning only. There is no
        # explicit CFG/unconditional or negative-prompt branch in this forward.
        with self.accelerator.autocast():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                list(prompts),
                prompt_2=None,
                prompt_3=None,
                do_classifier_free_guidance=False,
            )
        return prompt_embeds, pooled_prompt_embeds

    def _velocity_prediction(self, prompt_embeds: Tensor, pooled_prompt_embeds: Tensor):
        host = self._adapter_host(self.model)

        def predict(x_t: Tensor, flow_time: Tensor) -> Tensor:
            cond, pooled = repeat_endpoint_conditions(prompt_embeds, pooled_prompt_embeds, x_t.shape[0])
            timestep = flow_time.to(device=x_t.device, dtype=torch.float32) * 1000.0
            with self.accelerator.autocast():
                return host(
                    hidden_states=x_t,
                    timestep=timestep,
                    encoder_hidden_states=cond,
                    pooled_projections=pooled,
                    return_dict=False,
                )[0].float()

        return predict

    def _ensure_image_encoder(self) -> None:
        if self.distance_metric == "iem":
            raise RuntimeError("IEM does not use an image encoder")
        if self._image_encoder is not None:
            return

        model_field = IMAGE_DISTANCE_MODEL_FIELDS[self.distance_metric]
        model_id = str(getattr(self.config, model_field))
        if self.distance_metric == "tpips_overall":
            try:
                import tpips
            except ImportError as exc:
                raise RuntimeError(
                    "tpips_overall requires tpips==0.1.2 in a Torch>=2.7, "
                    "Transformers>=5, PEFT>=0.19 environment"
                ) from exc
            cache_dir = getattr(self.config, "tpips_cache_dir", None)
            self._image_encoder = tpips.load_model(
                "embedding",
                model_path=model_id,
                device="cpu",
                cache_dir=(None if cache_dir in (None, "") else str(cache_dir)),
            )
            self._image_encoder.requires_grad_(False).eval()
            return

        from transformers import (
            AutoImageProcessor,
            AutoModel,
            CLIPVisionModelWithProjection,
        )

        self._image_processor = AutoImageProcessor.from_pretrained(model_id)
        if self.distance_metric.startswith("clip_"):
            self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                model_id
            )
        else:
            self._image_encoder = AutoModel.from_pretrained(model_id)
        self._image_encoder.requires_grad_(False).eval()

    def _offload_image_encoder(self) -> None:
        if self._image_encoder is None:
            return
        self._image_encoder.to("cpu")
        if self.accelerator.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _image_embeddings_from_latents(self, latents: Tensor) -> Tensor:
        """Decode endpoints and return embeddings in the metric's required space."""

        if self.distance_metric == "iem":
            raise RuntimeError("IEM does not use decoded image embeddings")
        latents = torch.as_tensor(latents)
        if latents.ndim < 2 or latents.shape[0] < 1:
            raise ValueError("latent batch must be nonempty")

        self._ensure_image_encoder()
        self._image_encoder.to(self.accelerator.device).eval()
        features = []
        batch_size = int(
            self.config.tpips_batch_size
            if self.distance_metric == "tpips_overall"
            else self.config.feature_batch_size
        )
        for start in range(0, latents.shape[0], batch_size):
            latent_batch = latents[start : start + batch_size].to(
                device=self.accelerator.device
            )
            scaled = (
                latent_batch / self.pipe.vae.config.scaling_factor
                + self.pipe.vae.config.shift_factor
            )
            with self.accelerator.autocast():
                decoded = self.pipe.vae.decode(
                    scaled.to(self.pipe.vae.dtype),
                    return_dict=False,
                )[0]
            images = self.pipe.image_processor.postprocess(
                decoded,
                output_type="pt",
            )
            if self.distance_metric == "tpips_overall":
                embedding = self._image_encoder.embed(
                    images.float().cpu(),
                    factor="overall",
                    normalized=True,
                )
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                features.append(F.normalize(embedding.float(), p=2, dim=1))
                continue

            image_bytes = (
                images.mul(255)
                .round()
                .clamp(0, 255)
                .to(device="cpu", dtype=torch.uint8)
            )
            inputs = self._image_processor(
                images=[image for image in image_bytes],
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(self.accelerator.device)
            with self.accelerator.autocast():
                output = self._image_encoder(pixel_values=pixel_values)
            if self.distance_metric.startswith("clip_"):
                embedding = output.image_embeds
            else:
                embedding = output.last_hidden_state[:, 0]
            embedding = embedding.float()
            if self.distance_metric not in L2_DISTANCE_METRICS:
                embedding = F.normalize(embedding, p=2, dim=1)
            features.append(embedding)
        return torch.cat(features)

    @torch.no_grad()
    def refresh_references(self, epoch: int, excluded_prompts: Sequence[str]) -> None:
        if self.reference_prompt_mode != "diverse":
            raise RuntimeError(
                "the persistent reference reservoir is only used in diverse mode"
            )
        records = self.prompt_sampler.sample(
            self.config.reference_samples_per_epoch,
            seed=int(self.config.seed) + 1_000_003 + int(epoch),
            excluded_prompts=excluded_prompts,
        )
        local_records = records[self.accelerator.process_index :: self.accelerator.num_processes]
        local_latents = []
        batch_size = int(self.config.reference_batch_size)
        generator = torch.Generator(device=self.accelerator.device).manual_seed(
            int(self.config.seed)
            + 30_000_000
            + int(epoch) * self.accelerator.num_processes
            + self.accelerator.process_index
        )
        with self._reference_model():
            for start in range(0, len(local_records), batch_size):
                prompts = [record.text for record in local_records[start : start + batch_size]]
                with self.accelerator.autocast():
                    latents = self._reference_pipe(
                        prompts,
                        height=int(self.config.resolution),
                        width=int(self.config.resolution),
                        num_inference_steps=int(self.config.num_inference_steps),
                        guidance_scale=float(self.config.guidance_scale),
                        generator=generator,
                        output_type="latent",
                        return_dict=False,
                    )[0]
                local_latents.append(latents)

        local_latents_tensor = torch.cat(local_latents)
        local_prompt_ids = torch.tensor(
            [record.prompt_id for record in local_records],
            device=local_latents_tensor.device,
            dtype=torch.int64,
        )
        gathered_latents = self.accelerator.gather(local_latents_tensor)
        gathered_prompt_ids = self.accelerator.gather(local_prompt_ids)
        self.reservoir.update(
            gathered_latents,
            gathered_prompt_ids,
            seed=int(self.config.seed) + 10_000_000 + int(epoch),
        )

    def _reference_statistics(self, reference_latents: Tensor, reference_prompt_ids: Tensor, noise_table: Tensor):
        """Compute ``mu_omega`` and ``v_omega`` in equation 21."""

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        local_latents = reference_latents[rank::world_size]
        local_prompt_ids = reference_prompt_ids[rank::world_size]
        if local_latents.shape[0] < 1:
            raise ValueError("reference subset must contain at least one endpoint per process")

        local_count, local_sum_phi, local_sum_phi_squared_norm = 0, None, None
        batch_size = int(self.config.feature_batch_size)
        for start in range(0, local_latents.shape[0], batch_size):
            latent_batch = local_latents[start : start + batch_size].to(
                device=self.accelerator.device,
                dtype=torch.float32,
            )
            prompt_batch = [
                self.prompt_sampler.prompt_for_id(prompt_id)
                for prompt_id in local_prompt_ids[start : start + batch_size].tolist()
            ]
            prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompt_batch)
            features = iem_features(
                latent_batch,
                self._velocity_prediction(prompt_embeds, pooled_prompt_embeds),
                self.sigma_schedule,
                noise_table,
                level_batch_size=int(self.config.level_batch_size),
            )
            local_count, local_sum_phi, local_sum_phi_squared_norm = update_reference_sums(
                local_count,
                local_sum_phi,
                local_sum_phi_squared_norm,
                features,
            )

        reference_count = int(
            self.accelerator.reduce(
                torch.tensor(local_count, device=self.accelerator.device, dtype=torch.int64),
                reduction="sum",
            ).item()
        )
        # element-wise sum across all distributed processes
        sum_phi = self.accelerator.reduce(local_sum_phi, reduction="sum")
        sum_phi_squared_norm = self.accelerator.reduce(local_sum_phi_squared_norm, reduction="sum")

        mu_omega_64 = sum_phi / reference_count
        v_omega = (
            sum_phi_squared_norm / reference_count - mu_omega_64.square().sum()
        ).clamp_min(0)
        # clamp_min(0) is used to ensure that the variance is non-negative (v = E(y^2) - E(y)^2 >= 0)
        return reference_count, mu_omega_64.float(), v_omega

    @torch.no_grad()
    def _reference_cosine_mean(
        self,
        reference_latents: Tensor,
    ) -> tuple[int, Tensor]:
        """Compute the distributed mean of normalized reference embeddings."""

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        local_latents = reference_latents[rank::world_size]
        if local_latents.shape[0] < 1:
            raise ValueError(
                "reference subset must contain at least one endpoint per process"
            )

        local_count, local_sum = 0, None
        batch_size = int(self.config.feature_batch_size)
        for start in range(0, local_latents.shape[0], batch_size):
            features = self._image_embeddings_from_latents(
                local_latents[start : start + batch_size]
            )
            local_count, local_sum = update_reference_feature_sum(
                local_count,
                local_sum,
                features,
            )

        reference_count = int(
            self.accelerator.reduce(
                torch.tensor(local_count, device=self.accelerator.device),
                reduction="sum",
            ).item()
        )
        sum_features = self.accelerator.reduce(local_sum, reduction="sum")
        return reference_count, (sum_features / reference_count).float()

    @torch.no_grad()
    def _reference_cosine_features(
        self,
        reference_latents: Tensor,
    ) -> tuple[int, Tensor]:
        """Compute and gather every reference embedding for pairwise distances."""

        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        local_latents = reference_latents[rank::world_size]
        if local_latents.shape[0] < 1:
            raise ValueError(
                "reference subset must contain at least one endpoint per process"
            )

        local_features = []
        batch_size = int(self.config.feature_batch_size)
        for start in range(0, local_latents.shape[0], batch_size):
            local_features.append(
                self._image_embeddings_from_latents(
                    local_latents[start : start + batch_size]
                )
            )
        local_features = torch.cat(local_features)
        local_count = torch.tensor(
            [local_features.shape[0]],
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        counts = self.accelerator.gather(local_count)
        max_count = int(counts.max().item())

        padded_features = torch.zeros(
            max_count,
            local_features.shape[1],
            device=local_features.device,
            dtype=local_features.dtype,
        )
        padded_features[: local_features.shape[0]] = local_features
        valid = (
            torch.arange(
                max_count,
                device=local_features.device,
            )
            .lt(local_features.shape[0])
            .to(dtype=torch.uint8)
        )

        gathered_features = self.accelerator.gather(padded_features)
        gathered_valid = self.accelerator.gather(valid)
        reference_features = gathered_features[gathered_valid.bool()]
        reference_count = int(counts.sum().item())
        if reference_features.shape[0] != reference_count:
            raise RuntimeError(
                "distributed reference embedding gather returned the wrong count"
            )
        return reference_count, reference_features

    def _noise_table_assignments(self, candidate_count: int, group_size: int, epoch: int) -> Tensor:
        if candidate_count % group_size:
            raise ValueError("IEM candidate count must be divisible by the RAM prompt-group size")
        group_count = candidate_count // group_size
        generator = torch.Generator(device="cpu").manual_seed(
            int(self.config.seed)
            + 60_000_000
            + int(epoch) * self.accelerator.num_processes
            + self.accelerator.process_index
        )
        table_count = int(self.config.noise_table_count)
        group_tables = []
        previous_table = None
        while len(group_tables) < group_count:
            permutation = torch.randperm(table_count, generator=generator).tolist()
            if table_count > 1 and previous_table is not None and permutation[0] == previous_table:
                permutation[0], permutation[1] = permutation[1], permutation[0]
            group_tables.extend(permutation[: group_count - len(group_tables)])
            previous_table = group_tables[-1]
        group_tables = torch.tensor(group_tables, dtype=torch.long)
        return group_tables.repeat_interleave(group_size).to(self.accelerator.device)

    @staticmethod
    def _candidate_prompt_groups(
        candidate_count: int,
        candidate_prompts: Sequence[str],
        group_size: int,
    ) -> list[str]:
        """Validate flattened RAM prompt groups and return one prompt per group."""

        candidate_count = int(candidate_count)
        group_size = int(group_size)
        if group_size < 1 or candidate_count % group_size:
            raise ValueError(
                "IEM candidate count must be divisible by the RAM prompt-group size"
            )
        if len(candidate_prompts) != candidate_count:
            raise ValueError(
                "same-prompt IEM requires one prompt string per candidate latent"
            )

        group_prompts = []
        for start in range(0, candidate_count, group_size):
            prompt = str(candidate_prompts[start])
            if any(
                str(value) != prompt
                for value in candidate_prompts[start : start + group_size]
            ):
                raise ValueError(
                    "each RAM candidate group must contain exactly one prompt"
                )
            group_prompts.append(prompt)
        return group_prompts

    @torch.no_grad()
    def _score_same_prompt_cosine(
        self,
        candidate_latents: Tensor,
        candidate_prompt_embeds: Tensor,
        candidate_pooled_prompt_embeds: Tensor,
        candidate_prompts: Sequence[str],
        *,
        epoch: int,
        group_size: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Score candidates against same-prompt image-embedding references."""

        candidate_count = len(candidate_latents)
        if (
            candidate_prompt_embeds.shape[0] != candidate_count
            or candidate_pooled_prompt_embeds.shape[0] != candidate_count
        ):
            raise ValueError(
                "candidate latents and prompt-condition rows must be aligned"
            )
        group_prompts = self._candidate_prompt_groups(
            candidate_count,
            candidate_prompts,
            group_size,
        )
        scores = torch.empty(
            candidate_count,
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        references_per_prompt = int(self.config.reference_samples_per_prompt)
        reference_batch_size = int(self.config.reference_batch_size)
        reference_generation_seconds = 0.0
        reference_features_seconds = 0.0
        reference_cache_load_seconds = 0.0
        candidate_features_seconds = 0.0
        reference_selection_seconds = 0.0
        selected_reference_count = references_per_prompt
        cache_active = self._reference_cache_active(epoch)

        try:
            with self._reference_model():
                for group_index, prompt in enumerate(group_prompts):
                    candidate_start = group_index * int(group_size)
                    candidate_stop = candidate_start + int(group_size)
                    prompt_embed = candidate_prompt_embeds[
                        candidate_start : candidate_start + 1
                    ]
                    pooled_prompt_embed = candidate_pooled_prompt_embeds[
                        candidate_start : candidate_start + 1
                    ]
                    reference_count, sum_features = 0, None
                    reference_feature_batches = (
                        []
                        if (
                            self.reference_selection_mode == "nearest"
                            or self.distance_metric in L2_DISTANCE_METRICS
                        )
                        else None
                    )
                    cache_key = (
                        CLIP_KEY
                        if self.distance_metric.startswith("clip_")
                        else DINO_KEY
                    )
                    stage_start = synchronized_time(self.accelerator.device)
                    cached = self._cached_references(
                        epoch=epoch,
                        prompt=prompt,
                        tensor_keys=(cache_key,),
                    )
                    if cached is not None:
                        reference_features = cached[cache_key].to(
                            self.accelerator.device
                        ).float()
                        if self.distance_metric not in L2_DISTANCE_METRICS:
                            reference_features = F.normalize(
                                reference_features, p=2, dim=1
                            )
                        reference_cache_load_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )
                        stage_start = synchronized_time(
                            self.accelerator.device
                        )
                        candidate_features = self._image_embeddings_from_latents(
                            candidate_latents[candidate_start:candidate_stop]
                        )
                        candidate_features_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )
                        if self.distance_metric in L2_DISTANCE_METRICS:
                            if self.reference_selection_mode == "nearest":
                                group_scores, selected_reference_count = (
                                    mean_nearest_l2_distance(
                                        candidate_features,
                                        reference_features,
                                        self.nearest_reference_fraction,
                                    )
                                )
                            else:
                                group_scores = mean_pairwise_l2_distance(
                                    candidate_features, reference_features
                                )
                        elif self.reference_selection_mode == "nearest":
                            group_scores, selected_reference_count = (
                                mean_nearest_cosine_distance(
                                    candidate_features,
                                    reference_features,
                                    self.nearest_reference_fraction,
                                )
                            )
                        else:
                            group_scores = cosine_distance_from_reference_mean(
                                candidate_features,
                                reference_features.mean(dim=0),
                            )
                        scores[candidate_start:candidate_stop] = group_scores
                        continue
                    reference_seeds = [
                        same_prompt_reference_seed(
                            self.config.seed,
                            epoch,
                            prompt,
                            reference_index,
                        )
                        for reference_index in range(references_per_prompt)
                    ]
                    if len(set(reference_seeds)) != references_per_prompt:
                        raise RuntimeError(
                            "same-prompt reference initial-noise seeds "
                            "must be unique within each prompt group"
                        )

                    for reference_start in range(
                        0,
                        references_per_prompt,
                        reference_batch_size,
                    ):
                        reference_stop = min(
                            reference_start + reference_batch_size,
                            references_per_prompt,
                        )
                        batch_count = reference_stop - reference_start
                        reference_prompt_embeds = prompt_embed.expand(
                            (batch_count,) + prompt_embed.shape[1:]
                        )
                        reference_pooled_prompt_embeds = (
                            pooled_prompt_embed.expand(
                                (batch_count,) + pooled_prompt_embed.shape[1:]
                            )
                        )
                        generators = [
                            torch.Generator(
                                device=self.accelerator.device
                            ).manual_seed(reference_seed)
                            for reference_seed in reference_seeds[
                                reference_start:reference_stop
                            ]
                        ]

                        stage_start = synchronized_time(
                            self.accelerator.device
                        )
                        with self.accelerator.autocast():
                            reference_latents = self._reference_pipe(
                                prompt_embeds=reference_prompt_embeds,
                                pooled_prompt_embeds=(
                                    reference_pooled_prompt_embeds
                                ),
                                height=int(self.config.resolution),
                                width=int(self.config.resolution),
                                num_inference_steps=int(
                                    self.config.num_inference_steps
                                ),
                                guidance_scale=float(
                                    self.config.guidance_scale
                                ),
                                generator=generators,
                                output_type="latent",
                                return_dict=False,
                            )[0]
                        reference_generation_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )

                        stage_start = synchronized_time(
                            self.accelerator.device
                        )
                        reference_features = (
                            self._image_embeddings_from_latents(
                                reference_latents
                            )
                        )
                        if reference_feature_batches is not None:
                            reference_feature_batches.append(reference_features)
                            reference_count += reference_features.shape[0]
                        else:
                            reference_count, sum_features = (
                                update_reference_feature_sum(
                                    reference_count,
                                    sum_features,
                                    reference_features,
                                )
                            )
                        reference_features_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )

                    if reference_count != references_per_prompt:
                        raise RuntimeError(
                            "same-prompt reference generation returned "
                            "an unexpected number of endpoints"
                        )
                    stage_start = synchronized_time(self.accelerator.device)
                    candidate_features = self._image_embeddings_from_latents(
                        candidate_latents[candidate_start:candidate_stop]
                    )
                    candidate_features_seconds += (
                        synchronized_time(self.accelerator.device) - stage_start
                    )

                    if self.distance_metric in L2_DISTANCE_METRICS:
                        reference_features = torch.cat(reference_feature_batches)
                        if self.reference_selection_mode == "nearest":
                            stage_start = synchronized_time(
                                self.accelerator.device
                            )
                            group_scores, selected_reference_count = (
                                mean_nearest_l2_distance(
                                    candidate_features,
                                    reference_features,
                                    self.nearest_reference_fraction,
                                )
                            )
                            reference_selection_seconds += (
                                synchronized_time(self.accelerator.device)
                                - stage_start
                            )
                        else:
                            group_scores = mean_pairwise_l2_distance(
                                candidate_features,
                                reference_features,
                            )
                    elif reference_feature_batches is not None:
                        stage_start = synchronized_time(self.accelerator.device)
                        group_scores, selected_reference_count = (
                            mean_nearest_cosine_distance(
                                candidate_features,
                                torch.cat(reference_feature_batches),
                                self.nearest_reference_fraction,
                            )
                        )
                        reference_selection_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )
                    else:
                        reference_mean = (sum_features / reference_count).float()
                        group_scores = cosine_distance_from_reference_mean(
                            candidate_features,
                            reference_mean,
                        )
                    scores[candidate_start:candidate_stop] = group_scores
        finally:
            self._offload_image_encoder()

        global_group_count = int(
            self.accelerator.reduce(
                torch.tensor(
                    len(group_prompts),
                    device=self.accelerator.device,
                    dtype=torch.int64,
                ),
                reduction="sum",
            ).item()
        )
        prefix = self.distance_metric
        metrics = {
            f"{prefix}_reference_bank_size": 0.0,
            f"{prefix}_reference_subset_size": float(references_per_prompt),
            f"{prefix}_same_prompt_reference_groups": float(global_group_count),
            f"{prefix}_same_prompt_references_generated": float(
                0 if cache_active
                else global_group_count * references_per_prompt
            ),
            f"{prefix}_same_prompt_references_loaded": float(
                global_group_count * references_per_prompt
                if cache_active else 0
            ),
            f"timing/{prefix}_reference_refresh_seconds": (
                reference_generation_seconds
            ),
            f"timing/{prefix}_reference_cache_load_seconds": (
                reference_cache_load_seconds
            ),
            f"timing/{prefix}_reference_features_seconds": (
                reference_features_seconds
            ),
            f"timing/{prefix}_candidate_features_seconds": (
                candidate_features_seconds
            ),
        }
        if self.reference_selection_mode == "nearest":
            metrics.update(
                {
                    f"{prefix}_selected_reference_count": float(
                        selected_reference_count
                    ),
                    f"{prefix}_selected_reference_fraction": (
                        selected_reference_count / references_per_prompt
                    ),
                    f"timing/{prefix}_reference_selection_seconds": (
                        reference_selection_seconds
                    ),
                }
            )
        return scores, metrics

    @torch.no_grad()
    def _iem_objective_scores(
        self,
        features: Tensor,
        mu_omega: Tensor,
        v_omega: Tensor | float,
        signal_shape: Sequence[int],
    ) -> Tensor:
        """Evaluate the configured IEM objective with one shared contract."""

        return iem_objective_from_reference_statistics(
            features,
            mu_omega,
            v_omega,
            self.iem_objective,
            sigma_schedule=self.sigma_schedule,
            signal_dimension=math.prod(signal_shape),
            density_a1=self.density_a1,
            density_a2=self.density_a2,
        )

    @torch.no_grad()
    def _iem_scores_from_statistics(
        self,
        latents: Tensor,
        prompt_embeds: Tensor,
        pooled_prompt_embeds: Tensor,
        noise_table: Tensor,
        mu_omega: Tensor,
        v_omega: Tensor,
    ) -> Tensor:
        scores = []
        batch_size = int(self.config.feature_batch_size)
        for start in range(0, len(latents), batch_size):
            stop = min(start + batch_size, len(latents))
            features = iem_features(
                latents[start:stop],
                self._velocity_prediction(
                    prompt_embeds[start:stop],
                    pooled_prompt_embeds[start:stop],
                ),
                self.sigma_schedule,
                noise_table,
                level_batch_size=int(self.config.level_batch_size),
            )
            scores.append(
                self._iem_objective_scores(
                    features,
                    mu_omega,
                    v_omega,
                    latents.shape[1:],
                )
            )
        return torch.cat(scores)


    @torch.no_grad()
    def _score_same_prompt(
        self,
        candidate_latents: Tensor,
        candidate_prompt_embeds: Tensor,
        candidate_pooled_prompt_embeds: Tensor,
        candidate_prompts: Sequence[str],
        *,
        epoch: int,
        group_size: int,
    ) -> tuple[Tensor, dict[str, float]]:
        """Score each candidate group against fresh references of its prompt."""

        candidate_count = len(candidate_latents)
        if (
            candidate_prompt_embeds.shape[0] != candidate_count
            or candidate_pooled_prompt_embeds.shape[0] != candidate_count
        ):
            raise ValueError(
                "candidate latents and prompt-condition rows must be aligned"
            )
        group_prompts = self._candidate_prompt_groups(
            candidate_count,
            candidate_prompts,
            group_size,
        )
        assignments = self._noise_table_assignments(
            candidate_count,
            int(group_size),
            epoch,
        )
        signal_shape = tuple(candidate_latents.shape[1:])

        scores = torch.empty(
            candidate_count,
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        reference_generation_seconds = 0.0
        reference_cache_load_seconds = 0.0
        reference_statistics_seconds = 0.0
        candidate_features_seconds = 0.0
        references_per_prompt = int(self.config.reference_samples_per_prompt)
        reference_batch_size = int(self.config.reference_batch_size)
        feature_batch_size = int(self.config.feature_batch_size)
        cache_active = self._reference_cache_active(epoch)
        group_assignments = assignments[:: int(group_size)]

        with self._reference_model():
            for table_index in group_assignments.unique().tolist():
                noise_seed = iem_noise_table_seed(
                    self.config.seed,
                    epoch,
                    self.config.noise_table_count,
                    table_index,
                )
                noise_table = sample_iem_noise_table(
                    self.sigma_schedule,
                    signal_shape,
                    device=self.accelerator.device,
                    dtype=torch.float32,
                    seed=noise_seed,
                )
                noise_sha256 = iem_noise_sha256(noise_table)
                table_group_indices = torch.nonzero(
                    group_assignments == table_index,
                    as_tuple=False,
                ).flatten().tolist()
                for group_index in table_group_indices:
                    prompt = group_prompts[group_index]
                    candidate_start = group_index * int(group_size)
                    candidate_stop = candidate_start + int(group_size)
                    prompt_embed = candidate_prompt_embeds[
                        candidate_start : candidate_start + 1
                    ]
                    pooled_prompt_embed = candidate_pooled_prompt_embeds[
                        candidate_start : candidate_start + 1
                    ]

                    stage_start = synchronized_time(self.accelerator.device)
                    cached_statistics = self._cached_iem_statistics(
                        epoch=epoch,
                        prompt=prompt,
                        noise_seed=noise_seed,
                        noise_sha256=noise_sha256,
                        feature_dimension=int(self.config.num_steps) * math.prod(signal_shape),
                    )
                    if cached_statistics is not None:
                        reference_cache_load_seconds += (
                            synchronized_time(self.accelerator.device) - stage_start
                        )
                        mu_omega, v_omega = cached_statistics
                        stage_start = synchronized_time(self.accelerator.device)
                        scores[candidate_start:candidate_stop] = (
                            self._iem_scores_from_statistics(
                                candidate_latents[
                                    candidate_start:candidate_stop
                                ],
                                candidate_prompt_embeds[
                                    candidate_start:candidate_stop
                                ],
                                candidate_pooled_prompt_embeds[
                                    candidate_start:candidate_stop
                                ],
                                noise_table,
                                mu_omega,
                                v_omega,
                            )
                        )
                        candidate_features_seconds += (
                            synchronized_time(self.accelerator.device) - stage_start
                        )
                        continue

                    reference_count = 0
                    sum_phi = None
                    sum_phi_squared_norm = None
                    stage_start = synchronized_time(self.accelerator.device)
                    cached = self._cached_references(
                        epoch=epoch,
                        prompt=prompt,
                        tensor_keys=(LATENT_KEY,),
                    )
                    reference_cache_load_seconds += (
                        synchronized_time(self.accelerator.device) - stage_start
                    ) if cached is not None else 0.0
                    reference_seeds = [
                        same_prompt_reference_seed(
                            self.config.seed,
                            epoch,
                            prompt,
                            reference_index,
                        )
                        for reference_index in range(
                            references_per_prompt
                        )
                    ]
                    if len(set(reference_seeds)) != references_per_prompt:
                        raise RuntimeError(
                            "same-prompt reference initial-noise seeds "
                            "must be unique within each prompt group"
                        )
                    for reference_start in range(
                        0,
                        references_per_prompt,
                        reference_batch_size,
                    ):
                        reference_stop = min(
                            reference_start + reference_batch_size,
                            references_per_prompt,
                        )
                        batch_count = reference_stop - reference_start
                        reference_prompt_embeds = prompt_embed.expand(
                            (batch_count,) + prompt_embed.shape[1:]
                        )
                        reference_pooled_prompt_embeds = (
                            pooled_prompt_embed.expand(
                                (batch_count,)
                                + pooled_prompt_embed.shape[1:]
                            )
                        )
                        generators = [
                            torch.Generator(
                                device=self.accelerator.device
                            ).manual_seed(reference_seed)
                            for reference_seed in reference_seeds[
                                reference_start:reference_stop
                            ]
                        ]

                        stage_start = synchronized_time(
                            self.accelerator.device
                        )
                        if cached is not None:
                            reference_latents = cached[LATENT_KEY][
                                reference_start:reference_stop
                            ].to(self.accelerator.device)
                            reference_cache_load_seconds += (
                                synchronized_time(self.accelerator.device)
                                - stage_start
                            )
                        else:
                            with self.accelerator.autocast():
                                reference_latents = self._reference_pipe(
                                    prompt_embeds=reference_prompt_embeds,
                                    pooled_prompt_embeds=(
                                        reference_pooled_prompt_embeds
                                    ),
                                    height=int(self.config.resolution),
                                    width=int(self.config.resolution),
                                    num_inference_steps=int(
                                        self.config.num_inference_steps
                                    ),
                                    guidance_scale=float(
                                        self.config.guidance_scale
                                    ),
                                    generator=generators,
                                    output_type="latent",
                                    return_dict=False,
                                )[0]
                            reference_generation_seconds += (
                                synchronized_time(self.accelerator.device)
                                - stage_start
                            )

                        stage_start = synchronized_time(
                            self.accelerator.device
                        )
                        for feature_start in range(
                            0,
                            batch_count,
                            feature_batch_size,
                        ):
                            feature_stop = min(
                                feature_start + feature_batch_size,
                                batch_count,
                            )
                            features = iem_features(
                                reference_latents[
                                    feature_start:feature_stop
                                ],
                                self._velocity_prediction(
                                    reference_prompt_embeds[
                                        feature_start:feature_stop
                                    ],
                                    reference_pooled_prompt_embeds[
                                        feature_start:feature_stop
                                    ],
                                ),
                                self.sigma_schedule,
                                noise_table,
                                level_batch_size=int(
                                    self.config.level_batch_size
                                ),
                            )
                            (
                                reference_count,
                                sum_phi,
                                sum_phi_squared_norm,
                            ) = update_reference_sums(
                                reference_count,
                                sum_phi,
                                sum_phi_squared_norm,
                                features,
                            )
                        reference_statistics_seconds += (
                            synchronized_time(self.accelerator.device)
                            - stage_start
                        )

                    if reference_count != references_per_prompt:
                        raise RuntimeError(
                            "same-prompt reference generation returned "
                            "an unexpected number of endpoints"
                        )
                    mu_omega, v_omega = finalize_reference_statistics(
                        reference_count,
                        sum_phi,
                        sum_phi_squared_norm,
                    )

                    stage_start = synchronized_time(self.accelerator.device)
                    for candidate_batch_start in range(
                        candidate_start,
                        candidate_stop,
                        feature_batch_size,
                    ):
                        candidate_batch_stop = min(
                            candidate_batch_start + feature_batch_size,
                            candidate_stop,
                        )
                        candidate_features = iem_features(
                            candidate_latents[
                                candidate_batch_start:candidate_batch_stop
                            ],
                            self._velocity_prediction(
                                candidate_prompt_embeds[
                                    candidate_batch_start:candidate_batch_stop
                                ],
                                candidate_pooled_prompt_embeds[
                                    candidate_batch_start:candidate_batch_stop
                                ],
                            ),
                            self.sigma_schedule,
                            noise_table,
                            level_batch_size=int(
                                self.config.level_batch_size
                            ),
                        )
                        scores[
                            candidate_batch_start:candidate_batch_stop
                        ] = self._iem_objective_scores(
                            candidate_features,
                            mu_omega,
                            v_omega,
                            signal_shape,
                        )
                    candidate_features_seconds += (
                        synchronized_time(self.accelerator.device) - stage_start
                    )

        local_table_mask = torch.zeros(
            int(self.config.noise_table_count),
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        local_table_mask[assignments.unique()] = 1
        global_table_mask = self.accelerator.reduce(
            local_table_mask,
            reduction="sum",
        )
        global_group_count = int(
            self.accelerator.reduce(
                torch.tensor(
                    len(group_prompts),
                    device=self.accelerator.device,
                    dtype=torch.int64,
                ),
                reduction="sum",
            ).item()
        )
        metrics = {
            "iem_reference_bank_size": 0.0,
            "iem_reference_subset_size": float(references_per_prompt),
            "iem_noise_tables_used": float(
                (global_table_mask > 0).sum().item()
            ),
            "iem_same_prompt_reference_groups": float(global_group_count),
            "iem_same_prompt_references_generated": float(
                0 if cache_active
                else global_group_count * references_per_prompt
            ),
            "iem_same_prompt_references_loaded": float(
                global_group_count * references_per_prompt
                if cache_active else 0
            ),
            "timing/iem_reference_refresh_seconds": (
                reference_generation_seconds
            ),
            "timing/iem_reference_cache_load_seconds": (
                reference_cache_load_seconds
            ),
            "timing/iem_reference_statistics_seconds": (
                reference_statistics_seconds
            ),
            "timing/iem_candidate_features_seconds": (
                candidate_features_seconds
            ),
        }
        return scores.float(), metrics

    @torch.no_grad()
    def _score_diverse_cosine(
        self,
        candidate_latents: Tensor,
        reference_latents: Tensor,
        *,
        reference_refresh_seconds: float,
    ) -> tuple[Tensor, dict[str, float]]:
        """Score candidates against the diverse reservoir image embeddings."""

        reference_features_seconds = 0.0
        candidate_features_seconds = 0.0
        reference_selection_seconds = 0.0
        selected_reference_count = None
        try:
            stage_start = synchronized_time(self.accelerator.device)
            if (
                self.reference_selection_mode == "nearest"
                or self.distance_metric in L2_DISTANCE_METRICS
            ):
                reference_count, reference_features = (
                    self._reference_cosine_features(reference_latents)
                )
            else:
                reference_count, reference_mean = self._reference_cosine_mean(
                    reference_latents
                )
            reference_features_seconds = (
                synchronized_time(self.accelerator.device) - stage_start
            )

            stage_start = synchronized_time(self.accelerator.device)
            candidate_features = self._image_embeddings_from_latents(
                candidate_latents
            )
            if self.reference_selection_mode == "all":
                if self.distance_metric in L2_DISTANCE_METRICS:
                    scores = mean_pairwise_l2_distance(
                        candidate_features,
                        reference_features,
                    )
                else:
                    scores = cosine_distance_from_reference_mean(
                        candidate_features,
                        reference_mean,
                    )
            candidate_features_seconds = (
                synchronized_time(self.accelerator.device) - stage_start
            )

            if self.reference_selection_mode == "nearest":
                stage_start = synchronized_time(self.accelerator.device)
                if self.distance_metric in L2_DISTANCE_METRICS:
                    scores, selected_reference_count = mean_nearest_l2_distance(
                        candidate_features,
                        reference_features,
                        self.nearest_reference_fraction,
                    )
                else:
                    scores, selected_reference_count = (
                        mean_nearest_cosine_distance(
                            candidate_features,
                            reference_features,
                            self.nearest_reference_fraction,
                        )
                    )
                reference_selection_seconds = (
                    synchronized_time(self.accelerator.device) - stage_start
                )
        finally:
            self._offload_image_encoder()

        prefix = self.distance_metric
        metrics = {
            f"{prefix}_reference_bank_size": float(len(self.reservoir)),
            f"{prefix}_reference_subset_size": float(reference_count),
            f"timing/{prefix}_reference_refresh_seconds": (
                reference_refresh_seconds
            ),
            f"timing/{prefix}_reference_features_seconds": (
                reference_features_seconds
            ),
            f"timing/{prefix}_candidate_features_seconds": (
                candidate_features_seconds
            ),
        }
        if self.reference_selection_mode == "nearest":
            metrics.update(
                {
                    f"{prefix}_selected_reference_count": float(
                        selected_reference_count
                    ),
                    f"{prefix}_selected_reference_fraction": (
                        selected_reference_count / reference_count
                    ),
                    f"timing/{prefix}_reference_selection_seconds": (
                        reference_selection_seconds
                    ),
                }
            )
        return scores.float(), metrics

    @torch.no_grad()
    def score(
        self,
        candidate_latents: Tensor,
        candidate_prompt_embeds: Tensor,
        candidate_pooled_prompt_embeds: Tensor,
        candidate_prompts: Sequence[str],
        *,
        epoch: int,
        group_size: int,
        reference_excluded_prompts: Sequence[str] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Generate frozen-reference samples and return raw distance rewards."""

        if self.reference_prompt_mode == "same_prompt":
            score_method = (
                self._score_same_prompt
                if self.distance_metric == "iem"
                else self._score_same_prompt_cosine
            )
            return score_method(
                candidate_latents,
                candidate_prompt_embeds,
                candidate_pooled_prompt_embeds,
                candidate_prompts,
                epoch=epoch,
                group_size=group_size,
            )

        stage_start = synchronized_time(self.accelerator.device)
        self.refresh_references(
            epoch,
            excluded_prompts=(
                reference_excluded_prompts
                if reference_excluded_prompts is not None
                else candidate_prompts
            ),
        )
        reference_refresh_seconds = synchronized_time(self.accelerator.device) - stage_start
        reference_latents, reference_prompt_ids = self.reservoir.subset(
            self.config.reference_subset_size,
            seed=int(self.config.seed) + 20_000_000 + int(epoch),
        )
        if reference_latents.shape[0] < self.accelerator.num_processes:
            raise ValueError(
                "creativity reference subset is smaller than the process count"
            )

        if self.distance_metric != "iem":
            return self._score_diverse_cosine(
                candidate_latents,
                reference_latents,
                reference_refresh_seconds=reference_refresh_seconds,
            )

        assignments = self._noise_table_assignments(len(candidate_latents), int(group_size), epoch)
        scores = torch.empty(len(candidate_latents), device=self.accelerator.device, dtype=torch.float64)
        signal_shape = tuple(candidate_latents.shape[1:])
        reference_statistics_seconds = 0.0
        candidate_features_seconds = 0.0
        with self._reference_model():
            for table_index in range(int(self.config.noise_table_count)):
                noise_table = sample_iem_noise_table(
                    self.sigma_schedule,
                    signal_shape,
                    device=self.accelerator.device,
                    dtype=torch.float32,
                    seed=int(self.config.seed)
                    + 90_000_000
                    + int(epoch) * int(self.config.noise_table_count)
                    + table_index,
                )
                stage_start = synchronized_time(self.accelerator.device)
                reference_count, mu_omega, v_omega = self._reference_statistics(
                    reference_latents,
                    reference_prompt_ids,
                    noise_table,
                )
                reference_statistics_seconds += synchronized_time(self.accelerator.device) - stage_start
                rows = torch.nonzero(assignments == table_index, as_tuple=False).flatten()
                batch_size = int(self.config.feature_batch_size)
                stage_start = synchronized_time(self.accelerator.device)
                for start in range(0, rows.numel(), batch_size):
                    row_batch = rows[start : start + batch_size]
                    if row_batch.numel() == 0:
                        continue
                    features = iem_features(
                        candidate_latents.index_select(0, row_batch),
                        self._velocity_prediction(
                            candidate_prompt_embeds.index_select(0, row_batch),
                            candidate_pooled_prompt_embeds.index_select(0, row_batch),
                        ),
                        self.sigma_schedule,
                        noise_table,
                        level_batch_size=int(self.config.level_batch_size),
                    )
                    scores[row_batch] = self._iem_objective_scores(
                        features,
                        mu_omega,
                        v_omega,
                        signal_shape,
                    )
                candidate_features_seconds += synchronized_time(self.accelerator.device) - stage_start

        local_table_mask = torch.zeros(
            int(self.config.noise_table_count),
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        local_table_mask[assignments.unique()] = 1
        global_table_mask = self.accelerator.reduce(local_table_mask, reduction="sum")
        metrics = {
            "iem_reference_bank_size": float(len(self.reservoir)),
            "iem_reference_subset_size": float(reference_count),
            "iem_noise_tables_used": float((global_table_mask > 0).sum().item()),
            "timing/iem_reference_refresh_seconds": reference_refresh_seconds,
            "timing/iem_reference_statistics_seconds": reference_statistics_seconds,
            "timing/iem_candidate_features_seconds": candidate_features_seconds,
        }
        return scores.float(), metrics

"""Runtime adapters for using creativity rewards in DiffusionNFT."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import Sampler

from flow_grpo.creativity import BalancedPromptSampler, IEMReward


class DistributedPromptGroupBatchSampler(Sampler[list[int]]):
    """Keep every repeated prompt group complete and local to one process."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        group_size: int,
        num_groups: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        ram_aligned: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        self.num_groups = int(num_groups)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.ram_aligned = bool(ram_aligned)
        self.epoch = 0

        if self.num_groups > len(dataset):
            raise ValueError("num_groups cannot exceed the number of training prompts")
        if self.num_groups % self.num_replicas:
            raise ValueError("num_groups must be divisible by the process count")
        if self.group_size % self.batch_size:
            raise ValueError("group_size must be divisible by the sampling batch size")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank is outside the process range")

    def __len__(self) -> int:
        groups_per_rank = self.num_groups // self.num_replicas
        batches_per_group = self.group_size // self.batch_size
        return groups_per_rank * batches_per_group

    def __iter__(self):
        if self.ram_aligned:
            generator = random.Random(self.seed + self.epoch)
            group_indices = generator.sample(
                range(len(self.dataset)), self.num_groups
            )
            generator.shuffle(group_indices)
        else:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            group_indices = torch.randperm(
                len(self.dataset), generator=generator
            )[: self.num_groups].tolist()
        local_indices = group_indices[self.rank :: self.num_replicas]
        batches_per_group = self.group_size // self.batch_size
        for prompt_index in local_indices:
            for _ in range(batches_per_group):
                yield [prompt_index] * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def fixed_validation_seed(base_seed: int, prompt: str) -> int:
    """Match RAM's stable prompt-specific validation latent seed."""

    payload = f"ram-validation\0{int(base_seed)}\0{prompt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def validation_prompts(prompt_files, count: int, seed: int) -> list[str]:
    sampler = BalancedPromptSampler(prompt_files)
    return [record.text for record in sampler.sample(count, seed)]


class CreativityRewardSet:
    """Evaluate one or more RAM-compatible creativity distances synchronously."""

    def __init__(self, pipe, model, distributed_context, config, reference_generator):
        metrics = list(
            getattr(config, "distance_metrics", None)
            or [str(config.distance_metric)]
        )
        if len(metrics) != len(set(metrics)):
            raise ValueError("creativity.distance_metrics contains duplicates")
        self.weights = {
            metric: float(getattr(config, "reward_weights", {}).get(metric, 1.0))
            for metric in metrics
        }
        self.rewards = {}
        for metric in metrics:
            metric_config = copy.deepcopy(config)
            metric_config.distance_metric = metric
            if metric == "iem" and metric_config.reference_selection_mode == "nearest":
                metric_config.reference_selection_mode = "all"
            self.rewards[metric] = IEMReward(
                pipe,
                model,
                distributed_context,
                metric_config,
                reference_latent_generator=reference_generator,
            )

    def score(self, *args, **kwargs):
        component_scores = {}
        metrics = {}
        total = None
        for name, reward in self.rewards.items():
            scores, reward_metrics = reward.score(*args, **kwargs)
            component_scores[name] = scores
            weighted = scores * self.weights[name]
            total = weighted if total is None else total + weighted
            metrics.update(reward_metrics)
            metrics[f"reward_weight/{name}"] = self.weights[name]
        component_scores["avg"] = total
        return component_scores, metrics

    def state_dict(self) -> dict:
        return {name: reward.state_dict() for name, reward in self.rewards.items()}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != set(self.rewards):
            raise ValueError("checkpoint creativity metrics do not match the config")
        for name, reward_state in state.items():
            self.rewards[name].load_state_dict(reward_state)


def _cpu_adapter_state(model, adapter_name: str) -> dict[str, torch.Tensor]:
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    return {name: value.detach().cpu() for name, value in state.items()}


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints load on the rank CUDA device, but RNG APIs require CPU byte tensors.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


def save_training_checkpoint(
    save_dir,
    model,
    optimizer,
    scaler,
    ema,
    creativity_rewards,
    config,
    *,
    next_epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
) -> Path | None:
    """Atomically save all state needed to resume at an epoch boundary."""

    local_rng = _rng_state()
    rng_states = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_rng, rng_states, dst=0)
    else:
        rng_states = [local_rng]

    checkpoint_path = None
    if rank == 0:
        root = Path(save_dir) / "checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = root / f"checkpoint-epoch-{int(next_epoch):04d}"
        if checkpoint_path.exists() and (checkpoint_path / "_SUCCESS").is_file():
            raise FileExistsError(f"completed checkpoint already exists: {checkpoint_path}")
        staging = root / f".{checkpoint_path.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            payload = {
                "format_version": 1,
                "next_epoch": int(next_epoch),
                "global_step": int(global_step),
                "default_adapter": _cpu_adapter_state(model, "default"),
                "old_adapter": _cpu_adapter_state(model, "old"),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                "creativity": creativity_rewards.state_dict(),
                "rng_states": rng_states,
            }
            torch.save(payload, staging / "training_state.pt")
            model.save_pretrained(
                staging / "lora",
                selected_adapters=["default"],
                safe_serialization=True,
            )
            with (staging / "resolved_config.json").open("w", encoding="utf-8") as handle:
                json.dump(config.to_dict(), handle, indent=2, sort_keys=True, default=str)
            (staging / "_SUCCESS").write_text("ok\n", encoding="ascii")
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            os.replace(staging, checkpoint_path)
            latest_tmp = root / f".latest.tmp-{uuid.uuid4().hex}"
            latest_tmp.write_text(checkpoint_path.name + "\n", encoding="ascii")
            os.replace(latest_tmp, root / "latest")
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    if world_size > 1:
        dist.barrier()
    return checkpoint_path


def resolve_checkpoint(path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / "latest").is_file():
        checkpoint = checkpoint / (checkpoint / "latest").read_text().strip()
    elif (checkpoint / "checkpoints" / "latest").is_file():
        root = checkpoint / "checkpoints"
        checkpoint = root / (root / "latest").read_text().strip()
    if not (checkpoint / "_SUCCESS").is_file():
        raise FileNotFoundError(f"checkpoint is incomplete or missing: {checkpoint}")
    return checkpoint


def load_training_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    ema,
    creativity_rewards,
    *,
    device,
    rank: int,
) -> tuple[int, int, Path]:
    checkpoint = resolve_checkpoint(path)
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location=device,
        weights_only=False,
    )
    if int(state.get("format_version", -1)) != 1:
        raise ValueError("unsupported NFT creativity checkpoint format")
    set_peft_model_state_dict(model, state["default_adapter"], adapter_name="default")
    set_peft_model_state_dict(model, state["old_adapter"], adapter_name="old")
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state["ema"] is not None:
        ema.load_state_dict(state["ema"])
    creativity_rewards.load_state_dict(state["creativity"])
    _restore_rng_state(state["rng_states"][rank])
    model.set_adapter("default")
    return int(state["next_epoch"]), int(state["global_step"]), checkpoint

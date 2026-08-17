"""Persistent same-prompt reference data for creativity rewards.

Each cache entry contains the frozen baseline's endpoint latents, raw image
embeddings, and all-reference IEM sufficient statistics for one
``(epoch, prompt)`` pair. Files are published atomically so an interrupted
precompute job can safely resume without trusting partial data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


CACHE_SCHEMA_VERSION = 2
CACHE_SPEC_FILENAME = "spec.json"
CACHE_SUCCESS_FILENAME = "_SUCCESS"
LATENT_KEY = "latents"
CLIP_KEY = "clip_embeddings"
DINO_KEY = "dino_embeddings"
SEED_KEY = "seeds"
IEM_MEAN_KEY = "iem_mean"
IEM_VARIANCE_KEY = "iem_variance"
IEM_NOISE_SEED_KEY = "iem_noise_seed"
IEM_NOISE_SHA256_KEY = "iem_noise_sha256"
PER_REFERENCE_TENSOR_KEYS = (LATENT_KEY, CLIP_KEY, DINO_KEY, SEED_KEY)
IEM_TENSOR_KEYS = (
    IEM_MEAN_KEY, IEM_VARIANCE_KEY, IEM_NOISE_SEED_KEY, IEM_NOISE_SHA256_KEY
)
REQUIRED_TENSOR_KEYS = PER_REFERENCE_TENSOR_KEYS + IEM_TENSOR_KEYS


def canonical_json(value: object) -> str:
    """Return a stable JSON representation used for cache identities."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def cache_spec_sha256(spec: Mapping[str, object]) -> str:
    """Hash a cache specification independently of formatting."""

    return hashlib.sha256(canonical_json(dict(spec)).encode("utf-8")).hexdigest()


def prompt_cache_key(prompt: str) -> str:
    """Return the collision-resistant filename key for a prompt."""

    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def entry_path(cache_dir: str | Path, epoch: int, prompt: str) -> Path:
    """Return the safetensors path for one epoch/prompt reference cloud."""

    return (
        Path(cache_dir)
        / "entries"
        / f"epoch_{int(epoch):03d}"
        / f"{prompt_cache_key(prompt)}.safetensors"
    )


def _entry_metadata(
    *,
    spec_sha256: str,
    epoch: int,
    prompt: str,
    prompt_id: int,
) -> dict[str, str]:
    return {
        "schema_version": str(CACHE_SCHEMA_VERSION),
        "spec_sha256": str(spec_sha256),
        "epoch": str(int(epoch)),
        "prompt": str(prompt),
        "prompt_id": str(int(prompt_id)),
    }


def write_cache_spec(cache_dir: str | Path, spec: Mapping[str, object]) -> str:
    """Create or validate the immutable cache specification."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    spec_sha256 = cache_spec_sha256(spec)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "spec_sha256": spec_sha256,
        "spec": spec,
    }
    path = cache_dir / CACHE_SPEC_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(
                f"reference cache specification mismatch at {path}"
            )
        return spec_sha256

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return spec_sha256


def read_cache_spec(cache_dir: str | Path) -> dict:
    """Read and internally validate a cache specification."""

    path = Path(cache_dir) / CACHE_SPEC_FILENAME
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported reference cache schema at {path}")
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"reference cache has no valid specification: {path}")
    actual_sha256 = cache_spec_sha256(spec)
    if payload.get("spec_sha256") != actual_sha256:
        raise ValueError(f"reference cache specification hash is invalid: {path}")
    return payload


def _validated_metadata(
    path: Path,
    metadata: Mapping[str, str] | None,
    *,
    spec_sha256: str,
    epoch: int,
    prompt: str,
) -> Mapping[str, str]:
    metadata = metadata or {}
    expected = {
        "schema_version": str(CACHE_SCHEMA_VERSION),
        "spec_sha256": str(spec_sha256),
        "epoch": str(int(epoch)),
        "prompt": str(prompt),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"reference cache metadata mismatch in {path}: {mismatches}"
        )
    return metadata


def validate_cache_entry(
    cache_dir: str | Path,
    *,
    spec_sha256: str,
    epoch: int,
    prompt: str,
    reference_count: int,
) -> bool:
    """Return whether a complete entry with the expected identity exists."""

    path = entry_path(cache_dir, epoch, prompt)
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            _validated_metadata(
                path,
                handle.metadata(),
                spec_sha256=spec_sha256,
                epoch=epoch,
                prompt=prompt,
            )
            keys = set(handle.keys())
            if keys != set(REQUIRED_TENSOR_KEYS):
                return False
            for key in PER_REFERENCE_TENSOR_KEYS:
                if handle.get_slice(key).get_shape()[0] != int(reference_count):
                    return False
            if len(handle.get_slice(IEM_MEAN_KEY).get_shape()) != 1:
                return False
            if handle.get_slice(IEM_VARIANCE_KEY).get_shape() != [1]:
                return False
            if handle.get_slice(IEM_NOISE_SEED_KEY).get_shape() != [1]:
                return False
            if handle.get_slice(IEM_NOISE_SHA256_KEY).get_shape() != [32]:
                return False
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def write_cache_entry(
    cache_dir: str | Path,
    *,
    spec_sha256: str,
    epoch: int,
    prompt: str,
    prompt_id: int,
    latents: torch.Tensor,
    clip_embeddings: torch.Tensor,
    dino_embeddings: torch.Tensor,
    seeds: Sequence[int] | torch.Tensor,
    iem_mean: torch.Tensor,
    iem_variance: torch.Tensor | float,
    iem_noise_seed: int,
    iem_noise_sha256: str,
) -> Path:
    """Atomically publish one reference cloud in its storage dtypes."""

    latents = torch.as_tensor(latents).detach().cpu().to(torch.bfloat16).contiguous()
    clip_embeddings = (
        torch.as_tensor(clip_embeddings).detach().cpu().float().contiguous()
    )
    dino_embeddings = (
        torch.as_tensor(dino_embeddings).detach().cpu().float().contiguous()
    )
    seeds = torch.as_tensor(seeds, dtype=torch.int64, device="cpu").contiguous()
    iem_mean = torch.as_tensor(iem_mean).detach().cpu().float().contiguous()
    iem_variance = (
        torch.as_tensor(iem_variance, dtype=torch.float64, device="cpu")
        .reshape(1)
        .contiguous()
    )
    iem_noise_seed = torch.tensor(
        [int(iem_noise_seed)], dtype=torch.int64, device="cpu"
    )
    try:
        noise_digest = bytes.fromhex(str(iem_noise_sha256))
    except ValueError as exc:
        raise ValueError("IEM noise SHA-256 must be hexadecimal") from exc
    iem_noise_sha256 = torch.tensor(list(noise_digest), dtype=torch.uint8)
    reference_count = latents.shape[0]
    if reference_count < 1:
        raise ValueError("a reference cache entry must be nonempty")
    if any(
        tensor.shape[0] != reference_count
        for tensor in (clip_embeddings, dino_embeddings, seeds)
    ):
        raise ValueError("cached latents, embeddings, and seeds must be aligned")
    if clip_embeddings.ndim != 2 or dino_embeddings.ndim != 2:
        raise ValueError("cached CLIP and DINO embeddings must be matrices")
    if iem_mean.ndim != 1 or iem_mean.numel() < 1:
        raise ValueError("cached IEM mean must be a nonempty vector")
    if not torch.isfinite(iem_mean).all():
        raise ValueError("cached IEM mean must be finite")
    if not torch.isfinite(iem_variance).all() or (iem_variance < 0).any():
        raise ValueError("cached IEM variance must be finite and non-negative")
    if iem_noise_sha256.numel() != 32:
        raise ValueError("IEM noise SHA-256 must contain exactly 32 bytes")

    path = entry_path(cache_dir, epoch, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {
            LATENT_KEY: latents,
            CLIP_KEY: clip_embeddings,
            DINO_KEY: dino_embeddings,
            SEED_KEY: seeds,
            IEM_MEAN_KEY: iem_mean,
            IEM_VARIANCE_KEY: iem_variance,
            IEM_NOISE_SEED_KEY: iem_noise_seed,
            IEM_NOISE_SHA256_KEY: iem_noise_sha256,
        },
        temporary,
        metadata=_entry_metadata(
            spec_sha256=spec_sha256,
            epoch=epoch,
            prompt=prompt,
            prompt_id=prompt_id,
        ),
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


class ReferenceCacheReader:
    """Strict reader for a completed same-prompt reference cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_spec_sha256: str | None = None,
        require_complete: bool = True,
    ):
        self.cache_dir = Path(cache_dir).resolve()
        payload = read_cache_spec(self.cache_dir)
        self.spec = payload["spec"]
        self.spec_sha256 = payload["spec_sha256"]
        if (
            expected_spec_sha256 is not None
            and str(expected_spec_sha256) != self.spec_sha256
        ):
            raise ValueError(
                "reference cache hash does not match "
                "creativity.reference_cache_spec_sha256"
            )
        if require_complete and not (
            self.cache_dir / CACHE_SUCCESS_FILENAME
        ).is_file():
            raise ValueError(f"reference cache is not complete: {self.cache_dir}")

    def has_entry(self, *, epoch: int, prompt: str) -> bool:
        """Return whether this cache contains the requested epoch/prompt pair."""

        return entry_path(self.cache_dir, epoch, prompt).is_file()

    def load(
        self,
        *,
        epoch: int,
        prompt: str,
        tensor_keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Load selected tensors after validating entry identity."""

        unknown = set(tensor_keys) - set(REQUIRED_TENSOR_KEYS)
        if unknown:
            raise ValueError(f"unknown reference cache tensor keys: {unknown}")
        path = entry_path(self.cache_dir, epoch, prompt)
        with safe_open(path, framework="pt", device="cpu") as handle:
            _validated_metadata(
                path,
                handle.metadata(),
                spec_sha256=self.spec_sha256,
                epoch=epoch,
                prompt=prompt,
            )
            missing = set(tensor_keys) - set(handle.keys())
            if missing:
                raise ValueError(f"reference cache entry {path} is missing {missing}")
            return {key: handle.get_tensor(key) for key in tensor_keys}


"""Utilities for using a frozen LoRA as the RAM baseline model."""

import os
from pathlib import Path

from peft import PeftModel


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def resolve_baseline_lora_path(path):
    """Resolve either a PEFT adapter directory or a checkpoint containing ``lora/``.

    The returned directory is validated before the multi-billion-parameter base
    model is loaded, so missing or unreadable weights fail quickly.
    """
    if path is None or not str(path).strip():
        return None

    root = Path(path).expanduser().resolve()
    candidates = (root, root / "lora")
    adapter_dir = next((p for p in candidates if (p / ADAPTER_CONFIG_NAME).is_file()), None)
    if adapter_dir is None:
        checked = ", ".join(str(p / ADAPTER_CONFIG_NAME) for p in candidates)
        raise FileNotFoundError(f"Could not find a PEFT LoRA adapter config. Checked: {checked}")

    weight_path = next((adapter_dir / name for name in ADAPTER_WEIGHT_NAMES if (adapter_dir / name).is_file()), None)
    if weight_path is None:
        expected = ", ".join(str(adapter_dir / name) for name in ADAPTER_WEIGHT_NAMES)
        raise FileNotFoundError(f"Could not find baseline LoRA weights. Expected one of: {expected}")
    if not os.access(weight_path, os.R_OK):
        raise PermissionError(
            f"Baseline LoRA weights are not readable by the current user: {weight_path}. "
            "Grant read permission or copy the adapter to a readable directory."
        )
    return adapter_dir


def merge_frozen_baseline_lora(transformer, path):
    """Load ``path`` into ``transformer`` and merge it into frozen base weights.

    Merging before RAM adapters are attached is mathematically equivalent to
    stacking the additive LoRAs. It also leaves PEFT's adapter controls solely
    responsible for RAM's ``default``, ``old``, and ``evaluation`` adapters, so
    disabling adapters exposes exactly this frozen baseline.
    """
    adapter_dir = resolve_baseline_lora_path(path)
    if adapter_dir is None:
        return transformer, None

    baseline_model = PeftModel.from_pretrained(
        transformer,
        str(adapter_dir),
        is_trainable=False,
    )
    transformer = baseline_model.merge_and_unload(safe_merge=True)
    transformer.requires_grad_(False)
    return transformer, adapter_dir


def validate_no_cfg(scales):
    """Require guidance scale 1 for a CFG-distilled baseline LoRA."""
    invalid = {name: float(value) for name, value in scales.items() if float(value) != 1.0}
    if invalid:
        values = ", ".join(f"{name}={value:g}" for name, value in invalid.items())
        raise ValueError(
            f"A CFG-distilled baseline LoRA must run without classifier-free guidance; expected scale 1.0, got {values}"
        )

"""Load a compact RAM checkpoint as a frozen merged SD3 baseline."""

from __future__ import annotations

import math
from pathlib import Path

from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file


RAM_ADAPTER_NAMES = ("default", "old", "evaluation")
RAM_COMPACT_FILENAME = "ram_adapters.safetensors"
SD3_LORA_TARGET_MODULES = (
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
)


def resolve_ram_checkpoint(path: str | Path) -> Path:
    """Validate and return a complete RAM epoch checkpoint directory."""

    checkpoint = Path(path).expanduser().resolve()
    compact = checkpoint / RAM_COMPACT_FILENAME
    if not (checkpoint / "_SUCCESS").is_file():
        raise FileNotFoundError(f"RAM checkpoint is incomplete: {checkpoint}")
    if not compact.is_file() or compact.stat().st_size == 0:
        raise FileNotFoundError(f"Missing RAM adapter weights: {compact}")
    return checkpoint


def merge_ram_checkpoint_adapter(
    transformer,
    checkpoint_path: str | Path,
    *,
    adapter_name: str = "evaluation",
    rank: int = 32,
    alpha: int = 64,
    merge_scale: float = 1.0,
):
    """Merge one RAM adapter into ``transformer`` and remove PEFT wrappers.

    RAM stores policy, lagged-policy, and evaluation/EMA adapters in one compact
    state dict. Reconstructing all three preserves the checkpoint key contract;
    only the requested active adapter is merged into the frozen base weights.
    """

    if adapter_name not in RAM_ADAPTER_NAMES:
        raise ValueError(
            f"RAM adapter must be one of {RAM_ADAPTER_NAMES}; got {adapter_name!r}"
        )
    merge_scale = float(merge_scale)
    if not math.isfinite(merge_scale) or merge_scale < 0.0:
        raise ValueError(
            f"RAM adapter merge_scale must be finite and nonnegative; got {merge_scale}"
        )
    checkpoint = resolve_ram_checkpoint(checkpoint_path)
    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        init_lora_weights="gaussian",
        target_modules=list(SD3_LORA_TARGET_MODULES),
    )
    model = get_peft_model(transformer, lora_config)
    for name in RAM_ADAPTER_NAMES[1:]:
        model.add_adapter(name, lora_config)

    compact = load_file(str(checkpoint / RAM_COMPACT_FILENAME), device="cpu")
    model_keys = set(model.state_dict())
    unknown = sorted(set(compact) - model_keys)
    if unknown:
        raise ValueError(
            f"RAM checkpoint contains {len(unknown)} unknown tensors; "
            f"first key: {unknown[0]}"
        )
    expected = {
        key for key in model_keys if f".{adapter_name}." in key
    }
    provided = {
        key for key in compact if f".{adapter_name}." in key
    }
    if not expected or provided != expected:
        missing = sorted(expected - provided)
        raise ValueError(
            "RAM checkpoint does not exactly cover the selected adapter; "
            f"expected={len(expected)}, provided={len(provided)}, "
            f"first missing={missing[0] if missing else None}"
        )

    model.load_state_dict(compact, strict=False)
    model.set_adapter(adapter_name)
    scaled_layers = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            scaling[adapter_name] *= merge_scale
            scaled_layers += 1
    if not scaled_layers:
        raise RuntimeError(
            f"RAM adapter {adapter_name!r} has no PEFT scaling entries to merge"
        )
    # PEFT merges the active adapter when adapter_names is omitted. Omitting the
    # newer optional argument also keeps this compatible with our AirCC PEFT.
    merged = model.merge_and_unload(safe_merge=True)
    merged.requires_grad_(False)
    return merged, checkpoint

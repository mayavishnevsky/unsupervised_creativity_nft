"""Evaluate an NFT checkpoint and its frozen baseline on fixed creative probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import fields
from pathlib import Path

import torch
import wandb
from diffusers import StableDiffusion3Pipeline
from ml_collections import ConfigDict
from peft import LoraConfig, PeftModel
from PIL import Image

from flow_grpo.adapter_schedule import linear_decay_strengths
from flow_grpo.baseline_lora import merge_frozen_baseline_lora
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.evaluation_renoising import RENOISING_SEED_DOMAIN
from flow_grpo.nft_creativity_runtime import (
    resolve_checkpoint,
    validation_prompt_seeds,
    validation_prompts,
)
from flow_grpo.nft_validation import render_fixed_validation
from flow_grpo.ram_baseline import merge_ram_checkpoint_adapter


def _load_trainable_lora(model, adapter_dir: Path) -> PeftModel:
    """Load a standard LoRA saved by newer PEFT into an older PEFT runtime."""

    with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    supported_fields = {field.name for field in fields(LoraConfig)}
    unsupported = {
        key: value for key, value in raw_config.items() if key not in supported_fields
    }
    if not unsupported:
        return PeftModel.from_pretrained(
            model,
            adapter_dir,
            is_trainable=True,
        )

    # These newer PEFT fields do not affect this ordinary LoRA checkpoint.
    inert_compatibility_fields = {
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "lora_bias": False,
        "qalora_group_size": 16,
        "target_parameters": None,
        "trainable_token_indices": None,
        "use_qalora": False,
    }
    unsafe = {
        key: value
        for key, value in unsupported.items()
        if key not in inert_compatibility_fields
        or value != inert_compatibility_fields[key]
    }
    if unsafe:
        raise RuntimeError(
            "the installed PEFT version cannot represent adapter settings: "
            f"{unsafe}"
        )

    compatible_config = LoraConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in supported_fields
        }
    )
    print(
        "Loading standard LoRA after ignoring inert newer-PEFT metadata: "
        f"{sorted(unsupported)}",
        flush=True,
    )
    return PeftModel.from_pretrained(
        model,
        adapter_dir,
        is_trainable=True,
        config=compatible_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline-lora",
        default=None,
        help=(
            "Optional frozen baseline LoRA override. When omitted, the baseline "
            "recorded in the checkpoint's resolved config is reconstructed."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--seeds-per-prompt", type=int, default=1)
    parser.add_argument(
        "--renoising-repeats",
        type=int,
        default=0,
        help=(
            "Additional re-noise/denoise passes after each step where the "
            "trained NFT LoRA has nonzero strength. Evaluation only."
        ),
    )
    parser.add_argument(
        "--adapter-full-strength-steps",
        type=int,
        help="Initial denoising updates that use NFT LoRA strength 1.",
    )
    parser.add_argument(
        "--adapter-zero-strength-steps",
        type=int,
        help="Final denoising updates that use NFT LoRA strength 0.",
    )
    parser.add_argument(
        "--schedule",
        action="append",
        default=[],
        metavar="LABEL:FULL_STEPS:ZERO_STEPS",
        help=(
            "Render an additional named LoRA schedule. May be repeated; all "
            "schedules share one model load and the same prompt-specific latents."
        ),
    )
    parser.add_argument("--wandb-project", default="unsupervised_creativity_nft")
    parser.add_argument(
        "--wandb-entity",
        default="mayavishnevsky-tel-aviv-university",
    )
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-run-id",
        help="Append probe media to this existing W&B run instead of creating one.",
    )
    parser.add_argument(
        "--wandb-config-key",
        default=None,
        help=(
            "Optional unique W&B config key for the probe manifest. The legacy "
            "key is preserved when this is omitted."
        ),
    )
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def _nonempty_line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _parse_schedule(value: str) -> tuple[str, int, int]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"invalid schedule {value!r}; expected LABEL:FULL_STEPS:ZERO_STEPS"
        )
    label, full_steps_text, zero_steps_text = parts
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None:
        raise ValueError(f"invalid schedule label: {label!r}")
    try:
        full_steps = int(full_steps_text)
        zero_steps = int(zero_steps_text)
    except ValueError as error:
        raise ValueError(f"schedule steps must be integers: {value!r}") from error
    return label, full_steps, zero_steps


def _schedule_manifest(total_steps: int, label: str, full_steps: int, zero_steps: int):
    strengths = linear_decay_strengths(total_steps, full_steps, zero_steps)
    return {
        "label": label,
        "mode": "linear_decay",
        "creative_adapter": "default",
        "full_strength_steps": full_steps,
        "linear_decay_steps": total_steps - full_steps - zero_steps,
        "zero_strength_steps": zero_steps,
        "strengths": list(strengths),
    }


def _log_paired_prompt_grids(
    *,
    output_dir: Path,
    label: str,
    prompts: list[str],
    baseline_records: list[dict],
    creative_records: list[dict],
    seeds_per_prompt: int,
    global_step: int | None,
) -> list[dict]:
    """Save and log RAM-style top-baseline/bottom-creative prompt grids."""

    if len(baseline_records) != len(creative_records):
        raise RuntimeError("baseline and creative render counts differ")
    paired = {}
    for baseline, creative in zip(baseline_records, creative_records, strict=True):
        identity = ("index", "prompt_index", "seed_index", "prompt", "seed")
        if any(baseline[key] != creative[key] for key in identity):
            raise RuntimeError("baseline and creative prompt/latent records differ")
        paired.setdefault(baseline["prompt_index"], []).append((baseline, creative))

    grid_dir = output_dir / "comparisons" / label
    grid_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    manifest_records = []
    for prompt_index, prompt in enumerate(prompts):
        samples = sorted(paired[prompt_index], key=lambda pair: pair[0]["seed_index"])
        if len(samples) != seeds_per_prompt:
            raise RuntimeError(
                f"prompt {prompt_index} has {len(samples)} of "
                f"{seeds_per_prompt} expected seed pairs"
            )
        with Image.open(samples[0][0]["path"]) as image:
            width, height = image.size
        panel = Image.new("RGB", (width * seeds_per_prompt, height * 2))
        seeds = []
        for column, (baseline, creative) in enumerate(samples):
            with Image.open(baseline["path"]) as image:
                panel.paste(image.convert("RGB"), (column * width, 0))
            with Image.open(creative["path"]) as image:
                panel.paste(image.convert("RGB"), (column * width, height))
            seeds.append(baseline["seed"])
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
        grid_path = grid_dir / (
            f"{prompt_index:03d}_{prompt_hash}_{seeds_per_prompt}seeds_comparison.png"
        )
        panel.save(grid_path)
        caption = (
            "top=frozen baseline, bottom=creative; columns share seeds; "
            f"schedule={label}; seeds={seeds} | {prompt}"
        )
        panels.append(wandb.Image(panel, caption=caption))
        manifest_records.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seeds": seeds,
                "comparison_file": str(grid_path),
            }
        )
    payload = {f"validation/creative_probes_{label}_{seeds_per_prompt}seeds": panels}
    if global_step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=global_step)
    return manifest_records


@torch.no_grad()
def _apply_checkpoint_ema(model: PeftModel, checkpoint: Path) -> int:
    """Apply the EMA adapter weights used by in-training NFT validation."""

    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    ema_state = state.get("ema")
    if not ema_state or not ema_state.get("ema_parameters"):
        raise ValueError(f"checkpoint does not contain EMA weights: {checkpoint}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ema_parameters = ema_state["ema_parameters"]
    if len(trainable) != len(ema_parameters):
        raise ValueError(
            "EMA parameter count does not match the loaded adapter: "
            f"{len(ema_parameters)} checkpoint tensors vs {len(trainable)} model tensors"
        )
    for index, (parameter, ema_parameter) in enumerate(
        zip(trainable, ema_parameters, strict=True)
    ):
        if parameter.shape != ema_parameter.shape:
            raise ValueError(
                f"EMA tensor {index} has shape {tuple(ema_parameter.shape)}, "
                f"expected {tuple(parameter.shape)}"
            )
        parameter.copy_(ema_parameter.to(device=parameter.device, dtype=parameter.dtype))

    global_step = int(state["global_step"])
    del state
    return global_step


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    prompt_file = Path(args.prompt_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (checkpoint / "resolved_config.json").open(encoding="utf-8") as handle:
        config = ConfigDict(json.load(handle))
    config.pretrained.model = args.model or config.pretrained.model
    if args.baseline_lora is not None:
        config.baseline_lora_path = str(
            Path(args.baseline_lora).expanduser().resolve()
        )
    config.save_dir = str(output_dir)
    config.validation.prompt_files = [str(prompt_file)]
    config.validation.prompt_count = args.prompt_count or _nonempty_line_count(prompt_file)
    config.validation.batch_size = int(args.batch_size)
    config.validation.base_seed = int(args.base_seed)
    if args.seeds_per_prompt <= 0:
        raise ValueError("--seeds-per-prompt must be positive")
    if args.renoising_repeats < 0:
        raise ValueError("--renoising-repeats must be nonnegative")
    # EMA is applied directly below, before both rendering calls.
    config.train.ema = False
    if (args.adapter_full_strength_steps is None) != (
        args.adapter_zero_strength_steps is None
    ):
        raise ValueError(
            "--adapter-full-strength-steps and --adapter-zero-strength-steps "
            "must be provided together"
        )
    if args.schedule and args.adapter_full_strength_steps is not None:
        raise ValueError(
            "--schedule cannot be combined with the legacy adapter schedule arguments"
        )
    if args.seeds_per_prompt > 1 and not args.schedule:
        raise ValueError("multi-seed paired grids require at least one --schedule")

    total_steps = int(config.sample.eval_num_steps)
    schedule_specs = [_parse_schedule(value) for value in args.schedule]
    if len({label for label, _full, _zero in schedule_specs}) != len(schedule_specs):
        raise ValueError("schedule labels must be unique")
    schedule_manifests = [
        _schedule_manifest(total_steps, label, full_steps, zero_steps)
        for label, full_steps, zero_steps in schedule_specs
    ]
    adapter_schedule = None
    schedule_suffix = ""
    if args.adapter_full_strength_steps is not None:
        adapter_schedule = _schedule_manifest(
            total_steps,
            "creative",
            args.adapter_full_strength_steps,
            args.adapter_zero_strength_steps,
        )
        schedule_suffix = (
            f"_full{args.adapter_full_strength_steps}_"
            f"decay{adapter_schedule['linear_decay_steps']}_"
            f"zero{args.adapter_zero_strength_steps}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("creative-probe evaluation requires a CUDA GPU")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=dtype,
        local_files_only=bool(config.pretrained.local_files_only),
    )
    pipeline.transformer, baseline_adapter_dir = merge_frozen_baseline_lora(
        pipeline.transformer,
        getattr(config, "baseline_lora_path", None),
    )
    ram_baseline_checkpoint = getattr(config, "ram_baseline_checkpoint", None)
    merged_ram_checkpoint = None
    if ram_baseline_checkpoint:
        if baseline_adapter_dir is not None:
            raise ValueError(
                "RAM checkpoint merging and baseline_lora_path cannot both be enabled"
            )
        pipeline.transformer, merged_ram_checkpoint = merge_ram_checkpoint_adapter(
            pipeline.transformer,
            ram_baseline_checkpoint,
            adapter_name=str(
                getattr(config, "ram_baseline_adapter", "evaluation")
            ),
            rank=int(config.train.lora_rank),
            alpha=int(config.train.lora_alpha),
            merge_scale=float(
                getattr(config, "ram_baseline_merge_scale", 1.0)
            ),
        )
    pipeline.transformer = _load_trainable_lora(
        pipeline.transformer,
        checkpoint / "lora",
    )
    pipeline.transformer.set_adapter("default")
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(desc="Creative probes", dynamic_ncols=True)

    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    pipeline.transformer.to(device, dtype=dtype)
    global_step = _apply_checkpoint_ema(pipeline.transformer, checkpoint)
    for parameter in pipeline.transformer.parameters():
        parameter.requires_grad_(False)

    text_encoders = [
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    def encode_prompts(prompts):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                prompts,
                128,
            )
        return prompt_embeds.to(device), pooled_prompt_embeds.to(device)

    prompts = validation_prompts(
        config.validation.prompt_files,
        int(config.validation.prompt_count),
        int(getattr(config.validation, "prompt_seed", config.validation.base_seed)),
    )
    prompt_seed_groups = [
        validation_prompt_seeds(
            int(config.validation.base_seed), prompt, args.seeds_per_prompt
        )
        for prompt in prompts
    ]
    manifest = {
        "checkpoint": str(checkpoint),
        "baseline_lora": (
            str(baseline_adapter_dir) if baseline_adapter_dir is not None else None
        ),
        "ram_baseline_checkpoint": (
            str(merged_ram_checkpoint)
            if merged_ram_checkpoint is not None
            else None
        ),
        "ram_baseline_adapter": (
            str(getattr(config, "ram_baseline_adapter", "evaluation"))
            if merged_ram_checkpoint is not None
            else None
        ),
        "ram_baseline_merge_scale": (
            float(getattr(config, "ram_baseline_merge_scale", 1.0))
            if merged_ram_checkpoint is not None
            else None
        ),
        "base_seed": int(config.validation.base_seed),
        "resolution": int(config.resolution),
        "num_inference_steps": int(config.sample.eval_num_steps),
        "guidance_scale": 1.0,
        "solver": str(config.sample.solver),
        "noise_level": float(config.sample.noise_level),
        "weights": "ema",
        "global_step": global_step,
        "adapter_schedule": adapter_schedule,
        "adapter_schedules": schedule_manifests,
        "seeds_per_prompt": int(args.seeds_per_prompt),
        "prompt_seed_groups": prompt_seed_groups,
        "evaluation_renoising": (
            {
                "additional_passes_per_active_step": args.renoising_repeats,
                "active_when": "trained NFT LoRA strength is greater than zero",
                "seed_domain": RENOISING_SEED_DOMAIN,
            }
            if args.renoising_repeats
            else None
        ),
        "samples": [
            {
                "index": prompt_index * args.seeds_per_prompt + seed_index,
                "prompt_index": prompt_index,
                "seed_index": seed_index,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10],
                "latent_seed": seed,
            }
            for prompt_index, (prompt, seeds) in enumerate(
                zip(prompts, prompt_seed_groups, strict=True)
            )
            for seed_index, seed in enumerate(seeds)
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if args.wandb_run_id:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
            mode=args.wandb_mode,
            dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
        )
        config_key = args.wandb_config_key or "creative_probe_8seed_creative15"
        if args.renoising_repeats:
            config_key += f"_renoise{args.renoising_repeats}x"
        wandb.config.update(
            {config_key: manifest},
            allow_val_change=True,
        )
    else:
        run_id = hashlib.sha256(str(output_dir).encode("utf-8")).hexdigest()[:12]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            id=run_id,
            resume="allow",
            mode=args.wandb_mode,
            dir=os.environ.get("WANDB_DIR", "/tmp/wandb"),
            config=manifest,
        )
    wandb_step = None if args.wandb_run_id else global_step
    epoch_label = checkpoint.name.removeprefix("checkpoint-epoch-")
    baseline_records = render_fixed_validation(
        pipeline,
        encode_prompts,
        config,
        device,
        rank=0,
        world_size=1,
        global_step=global_step,
        ema=None,
        trainable_parameters=[],
        label="creative_probes_baseline",
        baseline=True,
        seeds_per_prompt=args.seeds_per_prompt,
        wandb_log=not bool(schedule_specs),
    )
    if schedule_specs:
        manifest["prompt_grids"] = {}
        for label, full_steps, zero_steps in schedule_specs:
            output_label = (
                f"{label}_renoise{args.renoising_repeats}x"
                if args.renoising_repeats
                else label
            )
            creative_records = render_fixed_validation(
                pipeline,
                encode_prompts,
                config,
                device,
                rank=0,
                world_size=1,
                global_step=global_step,
                ema=None,
                trainable_parameters=[],
                label=f"creative_probes_final_epoch_{epoch_label}_{output_label}",
                baseline=False,
                adapter_full_strength_steps=full_steps,
                adapter_zero_strength_steps=zero_steps,
                seeds_per_prompt=args.seeds_per_prompt,
                wandb_log=False,
                renoising_repeats=args.renoising_repeats,
            )
            manifest["prompt_grids"][output_label] = _log_paired_prompt_grids(
                output_dir=output_dir,
                label=output_label,
                prompts=prompts,
                baseline_records=baseline_records,
                creative_records=creative_records,
                seeds_per_prompt=args.seeds_per_prompt,
                global_step=wandb_step,
            )
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
    else:
        if args.renoising_repeats:
            schedule_suffix += f"_renoise{args.renoising_repeats}x"
        render_fixed_validation(
            pipeline,
            encode_prompts,
            config,
            device,
            rank=0,
            world_size=1,
            global_step=global_step,
            ema=None,
            trainable_parameters=[],
            label=f"creative_probes_final_epoch_{epoch_label}{schedule_suffix}",
            baseline=False,
            adapter_full_strength_steps=args.adapter_full_strength_steps,
            adapter_zero_strength_steps=args.adapter_zero_strength_steps,
            renoising_repeats=args.renoising_repeats,
        )
    wandb.finish()


if __name__ == "__main__":
    main()

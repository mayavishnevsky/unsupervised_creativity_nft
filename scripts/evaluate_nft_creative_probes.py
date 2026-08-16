"""Evaluate an NFT checkpoint and its frozen baseline on fixed creative probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import wandb
from diffusers import StableDiffusion3Pipeline
from ml_collections import ConfigDict
from peft import PeftModel

from flow_grpo.baseline_lora import merge_frozen_baseline_lora
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.nft_creativity_runtime import (
    fixed_validation_seed,
    resolve_checkpoint,
    validation_prompts,
)
from flow_grpo.nft_validation import render_fixed_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-lora", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="unsupervised_creativity_nft")
    parser.add_argument(
        "--wandb-entity",
        default="mayavishnevsky-tel-aviv-university",
    )
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def _nonempty_line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


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
    config.baseline_lora_path = str(Path(args.baseline_lora).expanduser().resolve())
    config.save_dir = str(output_dir)
    config.validation.prompt_files = [str(prompt_file)]
    config.validation.prompt_count = args.prompt_count or _nonempty_line_count(prompt_file)
    config.validation.batch_size = int(args.batch_size)
    config.validation.base_seed = int(args.base_seed)
    # EMA is applied directly below, before both rendering calls.
    config.train.ema = False

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
        config.baseline_lora_path,
    )
    pipeline.transformer = PeftModel.from_pretrained(
        pipeline.transformer,
        checkpoint / "lora",
        is_trainable=True,
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
        int(config.validation.base_seed),
    )
    manifest = {
        "checkpoint": str(checkpoint),
        "baseline_lora": str(baseline_adapter_dir),
        "base_seed": int(config.validation.base_seed),
        "resolution": int(config.resolution),
        "num_inference_steps": int(config.sample.eval_num_steps),
        "guidance_scale": 1.0,
        "solver": str(config.sample.solver),
        "noise_level": float(config.sample.noise_level),
        "weights": "ema",
        "global_step": global_step,
        "samples": [
            {
                "index": index,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10],
                "latent_seed": fixed_validation_seed(
                    int(config.validation.base_seed), prompt
                ),
            }
            for index, prompt in enumerate(prompts)
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

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
    epoch_label = checkpoint.name.removeprefix("checkpoint-epoch-")
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
        label="creative_probes_baseline",
        baseline=True,
    )
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
        label=f"creative_probes_final_epoch_{epoch_label}",
        baseline=False,
    )
    wandb.finish()


if __name__ == "__main__":
    main()

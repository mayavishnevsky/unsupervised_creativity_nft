"""Fixed-latent validation for NFT creativity training."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from PIL import Image

from flow_grpo.adapter_schedule import LoraAdapterScaler, linear_decay_strengths
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.nft_creativity_runtime import validation_prompt_seeds, validation_prompts


@torch.no_grad()
def render_fixed_validation(
    pipeline,
    encode_prompts,
    config,
    device,
    rank,
    world_size,
    global_step,
    ema,
    trainable_parameters,
    *,
    label: str,
    baseline: bool = False,
    adapter_full_strength_steps: int | None = None,
    adapter_zero_strength_steps: int | None = None,
    seeds_per_prompt: int = 1,
    wandb_log: bool = True,
):
    """Render fixed prompt/latent pairs and log a separate W&B panel."""

    if (adapter_full_strength_steps is None) != (adapter_zero_strength_steps is None):
        raise ValueError(
            "adapter_full_strength_steps and adapter_zero_strength_steps "
            "must be provided together"
        )
    if baseline and adapter_full_strength_steps is not None:
        raise ValueError("baseline validation cannot use an adapter strength schedule")
    seeds_per_prompt = int(seeds_per_prompt)
    if seeds_per_prompt <= 0:
        raise ValueError("seeds_per_prompt must be positive")
    adapter_strengths = None
    if adapter_full_strength_steps is not None:
        adapter_strengths = linear_decay_strengths(
            int(config.sample.eval_num_steps),
            adapter_full_strength_steps,
            adapter_zero_strength_steps,
        )

    if world_size > 1:
        dist.barrier()
    if rank != 0:
        if world_size > 1:
            dist.barrier()
        return []

    prompts = validation_prompts(
        config.validation.prompt_files,
        int(config.validation.prompt_count),
        int(
            getattr(config.validation, "prompt_seed", config.validation.base_seed)
        ),
    )
    prompt_seed_groups = [
        validation_prompt_seeds(
            int(config.validation.base_seed), prompt, seeds_per_prompt
        )
        for prompt in prompts
    ]
    all_seeds = [seed for seeds in prompt_seed_groups for seed in seeds]
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError("validation seeds must be unique across prompts")
    cases = [
        {
            "index": prompt_index * seeds_per_prompt + seed_index,
            "prompt_index": prompt_index,
            "seed_index": seed_index,
            "prompt": prompt,
            "seed": seed,
        }
        for prompt_index, (prompt, seeds) in enumerate(
            zip(prompts, prompt_seed_groups, strict=True)
        )
        for seed_index, seed in enumerate(seeds)
    ]
    model = pipeline.transformer
    if not baseline and config.train.ema and ema is not None:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
    model.set_adapter("default")
    adapter_context = model.disable_adapter() if baseline else nullcontext()
    adapter_scaler = (
        LoraAdapterScaler(model, "default")
        if adapter_strengths is not None
        else None
    )
    output_dir = Path(config.save_dir) / "validation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    images_to_log = []
    records = []
    amp_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16

    try:
        with adapter_context:
            batch_size = int(config.validation.batch_size)
            for start in range(0, len(cases), batch_size):
                batch_cases = cases[start : start + batch_size]
                batch_prompts = [case["prompt"] for case in batch_cases]
                prompt_embeds, pooled_prompt_embeds = encode_prompts(batch_prompts)
                latent_shape = (
                    model.config.in_channels,
                    int(config.resolution) // pipeline.vae_scale_factor,
                    int(config.resolution) // pipeline.vae_scale_factor,
                )
                initial_latents = torch.stack(
                    [
                        torch.randn(
                            latent_shape,
                            generator=torch.Generator(device=device).manual_seed(
                                case["seed"]
                            ),
                            device=device,
                            dtype=prompt_embeds.dtype,
                        )
                        for case in batch_cases
                    ]
                )
                with torch.autocast(
                    device_type="cuda",
                    enabled=config.mixed_precision in ("fp16", "bf16"),
                    dtype=amp_dtype,
                ):
                    images, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        latents=initial_latents,
                        num_inference_steps=int(config.sample.eval_num_steps),
                        guidance_scale=1.0,
                        output_type="pt",
                        height=int(config.resolution),
                        width=int(config.resolution),
                        noise_level=float(config.sample.noise_level),
                        deterministic=True,
                        solver=str(config.sample.solver),
                        model_type="sd3",
                        denoiser_step_callback=(
                            None
                            if adapter_scaler is None
                            else lambda step_index: adapter_scaler.set_strength(
                                adapter_strengths[step_index]
                            )
                        ),
                    )
                for case, image in zip(batch_cases, images, strict=True):
                    index = case["index"]
                    prompt = case["prompt"]
                    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
                    if seeds_per_prompt == 1:
                        filename = f"{index:03d}_{prompt_hash}.png"
                    else:
                        filename = (
                            f"{case['prompt_index']:03d}_{prompt_hash}_"
                            f"sample-{case['seed_index']:02d}_seed-{case['seed']}.png"
                        )
                    path = output_dir / filename
                    array = (
                        image.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                        * 255
                    ).round().astype(np.uint8)
                    Image.fromarray(array).save(path)
                    record = {**case, "path": str(path)}
                    records.append(record)
                    if wandb_log:
                        images_to_log.append(
                            wandb.Image(
                                str(path),
                                caption=(
                                    f"{case['prompt_index']:03d} | "
                                    f"sample {case['seed_index']} | seed {case['seed']} | "
                                    f"{prompt}"
                                ),
                            )
                        )
        if wandb_log:
            wandb.log({f"validation/{label}": images_to_log}, step=global_step)
    finally:
        if adapter_scaler is not None:
            adapter_scaler.restore()
        if not baseline and config.train.ema and ema is not None:
            ema.copy_temp_to(trainable_parameters)
        model.set_adapter("default")
    if world_size > 1:
        dist.barrier()
    return records

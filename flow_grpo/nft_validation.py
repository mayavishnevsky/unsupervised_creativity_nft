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

from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.nft_creativity_runtime import fixed_validation_seed, validation_prompts


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
):
    """Render fixed prompt/latent pairs and log a separate W&B panel."""

    if world_size > 1:
        dist.barrier()
    if rank != 0:
        if world_size > 1:
            dist.barrier()
        return

    prompts = validation_prompts(
        config.validation.prompt_files,
        int(config.validation.prompt_count),
        int(config.validation.base_seed),
    )
    model = pipeline.transformer
    if not baseline and config.train.ema and ema is not None:
        ema.copy_ema_to(trainable_parameters, store_temp=True)
    model.set_adapter("default")
    adapter_context = model.disable_adapter() if baseline else nullcontext()
    output_dir = Path(config.save_dir) / "validation" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    images_to_log = []
    amp_dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16

    try:
        with adapter_context:
            batch_size = int(config.validation.batch_size)
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
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
                                fixed_validation_seed(
                                    int(config.validation.base_seed), prompt
                                )
                            ),
                            device=device,
                            dtype=prompt_embeds.dtype,
                        )
                        for prompt in batch_prompts
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
                    )
                for offset, (prompt, image) in enumerate(zip(batch_prompts, images)):
                    index = start + offset
                    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
                    path = output_dir / f"{index:03d}_{prompt_hash}.png"
                    array = (
                        image.float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                        * 255
                    ).round().astype(np.uint8)
                    Image.fromarray(array).save(path)
                    images_to_log.append(
                        wandb.Image(str(path), caption=f"{index:03d} | {prompt}")
                    )
        wandb.log({f"validation/{label}": images_to_log}, step=global_step)
    finally:
        if not baseline and config.train.ema and ema is not None:
            ema.copy_temp_to(trainable_parameters)
        model.set_adapter("default")
    if world_size > 1:
        dist.barrier()

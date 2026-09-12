# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import hashlib
import datetime
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from functools import partial
import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.baseline_lora import merge_frozen_baseline_lora, validate_no_cfg
from flow_grpo.image_reward_runtime import StandardImageRewardSet
from flow_grpo.ram_baseline import merge_ram_checkpoint_adapter
from flow_grpo.candidate_diversity import (
    candidate_diversity_controls,
    candidate_diversity_seed,
    make_generator as make_diversity_generator,
    validate_candidate_diversity_sampling,
)
from flow_grpo.creativity import (
    BalancedPromptSampler,
    TorchDistributedContext,
    synchronized_time,
)
from flow_grpo.nft_validation import (
    log_paired_prompt_grids,
    render_fixed_validation,
)
from flow_grpo.nft_creativity_runtime import (
    CreativityRewardSet,
    DistributedPromptGroupBatchSampler,
    ReferenceModelState,
    load_training_checkpoint,
    save_training_checkpoint,
)
from flow_grpo.reference_model_schedule import (
    ReferenceModelUpdatePolicy,
    reference_adapter_context,
)
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train", prompt_files=None):
        if prompt_files is None:
            self.file_path = os.path.join(dataset, f"{split}.txt")
            self.file_paths = [self.file_path]
            with open(self.file_path, "r") as handle:
                self.prompts = [line.strip() for line in handle.readlines()]
            return

        self.file_paths = [str(path) for path in prompt_files]
        self.file_path = self.file_paths[0]
        self.prompts = []
        for path in self.file_paths:
            with open(path, "r", encoding="utf-8") as handle:
                self.prompts.extend(
                    line.strip() for line in handle if line.strip()
                )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas



def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()



def main(_):
    config = FLAGS.config
    creativity_enabled = bool(getattr(config.creativity, "enabled", True))
    reference_model_update = ReferenceModelUpdatePolicy.from_config(
        config.creativity
    )
    if not creativity_enabled and reference_model_update.enabled:
        raise ValueError("standard image rewards cannot update an IEM reference model")
    if reference_model_update.enabled and not config.use_lora:
        raise ValueError("rolling reference updates require LoRA training")
    diversity_method, diversity_config = validate_candidate_diversity_sampling(
        config
    )
    logger.info("Candidate diversity method: %s", diversity_method)

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    run_name = str(config.run_name or datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    run_name_object = [run_name if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(run_name_object, src=0)
    config.run_name = run_name_object[0]

    # W&B runtime files stay node-local; persistent artifacts use save_dir.
    if is_main_process(rank):
        wandb_dir = os.environ.get("WANDB_DIR", "/tmp/wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        run_id = os.environ.get("WANDB_RUN_ID") or hashlib.sha256(
            os.path.abspath(config.save_dir).encode("utf-8")
        ).hexdigest()[:12]
        wandb.init(
            project=str(config.wandb_project),
            entity=str(config.wandb_entity),
            name=config.run_name,
            id=run_id,
            resume="allow",
            mode=str(config.wandb_mode),
            config=config.to_dict(),
            dir=wandb_dir,
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=(mixed_precision_dtype == torch.float16))

    # Merge the CFG-distilled baseline before adding NFT adapters.
    validate_no_cfg(
        {
            "sample.guidance_scale": config.sample.guidance_scale,
            "creativity.guidance_scale": config.creativity.guidance_scale,
        }
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=mixed_precision_dtype or torch.float32,
        local_files_only=bool(getattr(config.pretrained, "local_files_only", False)),
    )
    pipeline.transformer, baseline_adapter_dir = merge_frozen_baseline_lora(
        pipeline.transformer, config.baseline_lora_path
    )
    logger.info("Merged frozen baseline LoRA from %s", baseline_adapter_dir)
    ram_baseline_checkpoint = getattr(config, "ram_baseline_checkpoint", None)
    if ram_baseline_checkpoint:
        if baseline_adapter_dir is not None:
            raise ValueError(
                "RAM checkpoint merging and baseline_lora_path cannot both be enabled"
            )
        pipeline.transformer, ram_checkpoint = merge_ram_checkpoint_adapter(
            pipeline.transformer,
            ram_baseline_checkpoint,
            adapter_name=str(getattr(config, "ram_baseline_adapter", "evaluation")),
            rank=int(config.train.lora_rank),
            alpha=int(config.train.lora_alpha),
            merge_scale=float(getattr(config, "ram_baseline_merge_scale", 1.0)),
        )
        logger.info(
            "Merged frozen RAM %s adapter from %s at strength %.6g",
            str(getattr(config, "ram_baseline_adapter", "evaluation")),
            ram_checkpoint,
            float(getattr(config, "ram_baseline_merge_scale", 1.0)),
        )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=int(config.train.lora_rank), lora_alpha=int(config.train.lora_alpha), init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        if reference_model_update.enabled:
            transformer.add_adapter(
                reference_model_update.adapter_name,
                transformer_lora_config,
            )
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    pipeline.transformer = transformer
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    reference_transformer_parameters = None
    if reference_model_update.enabled:
        transformer_ddp.module.set_adapter(
            reference_model_update.adapter_name
        )
        reference_transformer_parameters = [
            parameter
            for parameter in transformer_ddp.module.parameters()
            if parameter.requires_grad
        ]
        for parameter in reference_transformer_parameters:
            parameter.data.zero_()
        transformer_ddp.module.set_adapter("default")
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    prompt_sampling_mode = str(
        getattr(config.creativity, "candidate_prompt_sampling_mode", "independent_epoch")
    )
    candidate_prompt_sampler = None
    if config.prompt_fn == "general_ocr":
        if prompt_sampling_mode == "no_repeat_cycle":
            candidate_prompt_sampler = BalancedPromptSampler(
                config.creativity.candidate_prompt_files
            )
            train_dataset = TextPromptDataset(
                config.dataset,
                "train",
                prompt_files=config.creativity.candidate_prompt_files,
            )
            expected_prompts = [
                candidate_prompt_sampler.prompt_for_id(index)
                for index in range(len(candidate_prompt_sampler))
            ]
            if train_dataset.prompts != expected_prompts:
                raise ValueError(
                    "candidate prompt files do not align with NFT dataset indices"
                )
        else:
            train_dataset = TextPromptDataset(config.dataset, "train")
    elif config.prompt_fn == "geneval":
        if prompt_sampling_mode == "no_repeat_cycle":
            raise ValueError("no_repeat_cycle currently requires text prompt files")
        train_dataset = GenevalPromptDataset(config.dataset, "train")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedPromptGroupBatchSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        group_size=config.sample.num_image_per_prompt,
        num_groups=config.sample.num_prompt_groups,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
        ram_aligned=bool(
            getattr(config.sample, "ram_aligned_prompt_sampling", False)
        ),
        prompt_sampler=candidate_prompt_sampler,
        sampling_mode=prompt_sampling_mode,
        source_weights=getattr(
            config.creativity,
            "candidate_prompt_source_weights",
            None,
        ),
    )
    if len(train_sampler) != int(config.sample.num_batches_per_epoch):
        raise ValueError(
            "sample.num_batches_per_epoch does not match prompt-group geometry: "
            f"configured={config.sample.num_batches_per_epoch}, actual={len(train_sampler)}"
        )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )


    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    def encode_prompt_batch(prompts):
        return compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )

    @torch.no_grad()
    def generate_reference_latents(*args, **kwargs):
        kwargs.update(
            {
                "output_type": "latent",
                "deterministic": True,
                "solver": str(config.sample.solver),
                "noise_level": float(config.sample.noise_level),
                "model_type": "sd3",
            }
        )
        return pipeline_with_logprob(pipeline, *args, **kwargs)[0]

    if creativity_enabled:
        distributed_context = TorchDistributedContext(device, mixed_precision_dtype)
        creativity_rewards = CreativityRewardSet(
            pipeline,
            transformer_ddp.module,
            distributed_context,
            config.creativity,
            generate_reference_latents,
            reference_adapter_name=reference_model_update.adapter_name,
        )
    else:
        creativity_rewards = StandardImageRewardSet(
            pipeline,
            device,
            config.reward_fn,
            decode_batch_size=int(config.reward_decode_batch_size),
        )

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    first_epoch = 0
    global_step = 0
    reference_model_state = ReferenceModelState()
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.9,
            update_step_interval=1,
            device=device,
        )

    if config.resume_from:
        first_epoch, global_step, resumed_checkpoint = load_training_checkpoint(
            config.resume_from,
            transformer_ddp.module,
            optimizer,
            scaler,
            ema,
            creativity_rewards,
            device=device,
            rank=rank,
            reference_model_state=(
                reference_model_state
                if reference_model_update.enabled
                else None
            ),
            reference_adapter_name=reference_model_update.adapter_name,
        )
        logger.info(
            "Resumed from %s at epoch %d",
            resumed_checkpoint,
            first_epoch,
        )
    if reference_model_update.enabled:
        expected_reference_epoch = (
            reference_model_update.expected_reference_epoch(
                first_epoch,
                int(config.num_epochs),
            )
        )
        if reference_model_state.epoch != expected_reference_epoch:
            raise ValueError(
                "checkpoint reference snapshot is inconsistent with its "
                f"next epoch: saved={reference_model_state.epoch}, "
                f"expected={expected_reference_epoch}"
            )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    optimizer.zero_grad()

    if not config.resume_from:
        for src_param, tgt_param in zip(
            transformer_trainable_parameters,
            old_transformer_trainable_parameters,
            strict=True,
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param

    if first_epoch == 0 and config.eval_before_training and not config.debug:
        render_fixed_validation(
            pipeline,
            encode_prompt_batch,
            config,
            device,
            rank,
            world_size,
            global_step,
            ema,
            transformer_trainable_parameters,
            label="baseline",
            baseline=True,
        )

    for epoch in range(first_epoch, config.num_epochs):
        train_sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)

        pipeline.transformer.eval()
        samples_data_list = []
        candidate_prompts = []
        sampling_start = synchronized_time(device)

        for sample_batch_index in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            prompts, prompt_metadata = next(train_iter)
            del prompt_metadata
            prompt_embeds, pooled_prompt_embeds = encode_prompt_batch(prompts)
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            diversity_generator = None
            if diversity_method == "cads":
                diversity_generator = make_diversity_generator(
                    device,
                    candidate_diversity_seed(
                        config.seed,
                        epoch,
                        rank,
                        sample_batch_index * config.sample.train_batch_size,
                    ),
                )

            transformer_ddp.module.set_adapter("old")
            try:
                with candidate_diversity_controls(
                    pipeline,
                    diversity_method,
                    diversity_config,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    prompts,
                    diversity_generator,
                ) as diversity_kwargs:
                    with torch_autocast(
                        enabled=enable_amp,
                        dtype=mixed_precision_dtype,
                    ):
                        with torch.no_grad():
                            latent_endpoints, _, _ = pipeline_with_logprob(
                                pipeline,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                num_inference_steps=config.sample.num_steps,
                                guidance_scale=config.sample.guidance_scale,
                                output_type="latent",
                                height=config.resolution,
                                width=config.resolution,
                                noise_level=config.sample.noise_level,
                                deterministic=config.sample.deterministic,
                                solver=config.sample.solver,
                                model_type="sd3",
                                **diversity_kwargs,
                            )
            finally:
                transformer_ddp.module.set_adapter("default")
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)
            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate(
                        [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
                    ),
                    "latents_clean": latent_endpoints,
                }
            )
            candidate_prompts.extend(prompts)

        sampling_seconds = synchronized_time(device) - sampling_start
        collated_samples = {
            key: torch.cat([sample[key] for sample in samples_data_list], dim=0)
            for key in samples_data_list[0]
        }

        reward_start = synchronized_time(device)
        reward_components, creativity_metrics = creativity_rewards.score(
            collated_samples["latents_clean"],
            collated_samples["prompt_embeds"],
            collated_samples["pooled_prompt_embeds"],
            candidate_prompts,
            epoch=epoch,
            group_size=int(config.sample.num_image_per_prompt),
        )
        reward_seconds = synchronized_time(device) - reward_start
        collated_samples["rewards"] = reward_components

        metric_values = {
            "timing/sampling_seconds": float(sampling_seconds),
            "timing/reward_total_seconds": float(reward_seconds),
            **creativity_metrics,
        }
        metric_names = sorted(metric_values)
        metric_tensor = torch.tensor(
            [metric_values[name] for name in metric_names],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
        if is_main_process(rank):
            wandb.log(
                {name: metric_tensor[index].item() for index, name in enumerate(metric_names)},
                step=global_step,
            )

        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Frozen reference-model part
                            if config.use_lora:
                                with reference_adapter_context(
                                    transformer_ddp.module,
                                    reference_model_update.adapter_name,
                                ):
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        completed_epoch = epoch + 1
        reference_model_updated = reference_model_update.should_update(
            completed_epoch,
            int(config.num_epochs),
            reference_model_state.epoch,
        )
        if reference_model_updated:
            with torch.no_grad():
                for source, target in zip(
                    transformer_trainable_parameters,
                    reference_transformer_parameters,
                    strict=True,
                ):
                    target.data.copy_(source.detach().data)
            reference_model_state.epoch = completed_epoch
            creativity_rewards.on_reference_model_updated()
            transformer_ddp.module.set_adapter("default")
            if world_size > 1:
                dist.barrier()
            logger.info(
                "Updated frozen reference model after epoch %d",
                completed_epoch,
            )
        if reference_model_update.enabled and is_main_process(rank):
            wandb.log(
                {
                    "epoch": epoch,
                    "reference_model/epoch": reference_model_state.epoch,
                    "reference_model/updated": int(reference_model_updated),
                }
            )
        if completed_epoch % int(config.save_freq) == 0 and not config.debug:
            transformer_ddp.module.set_adapter("default")
            checkpoint_path = save_training_checkpoint(
                config.save_dir,
                transformer_ddp.module,
                optimizer,
                scaler,
                ema,
                creativity_rewards,
                config,
                next_epoch=completed_epoch,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                reference_model_state=(
                    reference_model_state
                    if reference_model_update.enabled
                    else None
                ),
                reference_adapter_name=reference_model_update.adapter_name,
            )
            if is_main_process(rank):
                logger.info("Saved complete checkpoint to %s", checkpoint_path)

        if completed_epoch % int(config.eval_freq) == 0 and not config.debug:
            render_fixed_validation(
                pipeline,
                encode_prompt_batch,
                config,
                device,
                rank,
                world_size,
                global_step,
                ema,
                transformer_trainable_parameters,
                label=f"epoch_{completed_epoch:04d}",
                baseline=False,
            )

    probe_config = getattr(config, "creative_probes", None)
    if (
        probe_config is not None
        and bool(getattr(probe_config, "enabled", False))
        and not config.debug
    ):
        saved_prompt_files = list(config.validation.prompt_files)
        saved_prompt_count = int(config.validation.prompt_count)
        saved_batch_size = int(config.validation.batch_size)
        prompt_file = str(probe_config.prompt_file)
        with open(prompt_file, encoding="utf-8") as handle:
            prompt_count = sum(bool(line.strip()) for line in handle)
        try:
            config.validation.prompt_files = [prompt_file]
            config.validation.prompt_count = prompt_count
            config.validation.batch_size = int(probe_config.batch_size)
            seeds_per_prompt = int(probe_config.seeds_per_prompt)
            ram_style_grids = bool(
                getattr(probe_config, "ram_style_paired_grids", False)
            )
            baseline_records = render_fixed_validation(
                pipeline,
                encode_prompt_batch,
                config,
                device,
                rank,
                world_size,
                global_step,
                ema,
                transformer_trainable_parameters,
                label=f"creative_probes_final_baseline_{seeds_per_prompt}seeds",
                baseline=True,
                seeds_per_prompt=seeds_per_prompt,
                wandb_log=not ram_style_grids,
            )
            creative_records = render_fixed_validation(
                pipeline,
                encode_prompt_batch,
                config,
                device,
                rank,
                world_size,
                global_step,
                ema,
                transformer_trainable_parameters,
                label=(
                    f"creative_probes_final_epoch_{int(config.num_epochs):04d}_"
                    f"{seeds_per_prompt}seeds"
                ),
                baseline=False,
                seeds_per_prompt=seeds_per_prompt,
                wandb_log=not ram_style_grids,
            )
            if ram_style_grids and is_main_process(rank):
                log_paired_prompt_grids(
                    output_dir=os.path.join(
                        config.save_dir, "validation", "creative_probes_ram_style"
                    ),
                    label=f"final_epoch_{int(config.num_epochs):04d}",
                    baseline_records=baseline_records,
                    creative_records=creative_records,
                    seeds_per_prompt=seeds_per_prompt,
                    global_step=global_step,
                )
        finally:
            config.validation.prompt_files = saved_prompt_files
            config.validation.prompt_count = saved_prompt_count
            config.validation.batch_size = saved_batch_size

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)

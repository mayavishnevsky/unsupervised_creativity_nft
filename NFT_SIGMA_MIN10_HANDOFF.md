# NFT sigma-min-10 experiment handoff

## Goal

Reproduce W&B run `d0e7a9f73f24` with exactly one training-configuration
change:

```text
creativity.sigma_min: 0.009 -> 10
```

There is no reward-multiplier change in this experiment. NFT does not use
RAM's `reward_multiplier`; the original NFT controls remain:

```text
config.beta = 0.1
config.train.beta = 0.25
```

The prepared two-GPU launcher is:

```text
experiment_scripts/run_nft_sigma_min10_aircc_cycle3_b200_2gpu.sbatch
```

The immutable source used by the submitted AirCC job is Git revision
`8d9e61b3eafcd44b303debd4682183152f15dab9`.

## Exact experiment configuration

The run keeps the following settings from `d0e7a9f73f24`:

```text
model: stabilityai/stable-diffusion-3.5-medium
baseline LoRA: CFG-distillation checkpoint 12
resolution: 512
epochs: 20
candidate generation steps: 25
evaluation steps: 40
candidate/evaluation guidance scale: 1.0
candidates per prompt: 24
prompt groups per epoch: 48
candidate prompt sampling: no_repeat_cycle
prompt-source weights: 0.50 / 0.25 / 0.25
reward: -log p(x) + 0.5 * (-g(x))
reference mode: same_prompt
references per candidate prompt: 256
reference selection: all
sigma range: 10 to 1000
IEM integration levels: 64
level_batch_size: 4
feature_batch_size: 16
NFT policy coefficient (config.beta): 0.1
KL coefficient (config.train.beta): 0.25
LoRA rank/alpha: 32 / 64
mixed precision: bf16
save frequency: every epoch
validation frequency: every epoch
baseline validation before training: disabled
```

Prompt sources, in order, are:

```text
dataset/hpdv3_plusplus/train.txt                  weight 0.50
dataset/creative_chat/creative_chat_train.txt    weight 0.25
dataset/partiprompts_basic/train.txt              weight 0.25
```

Validation uses the first 30 prompts selected from:

```text
dataset/partiprompts_basic/test.txt
dataset/partiprompts_imagination/test.txt
```

## Required environment and assets

Use the repository's DiffusionNFT environment. On the original systems this
is either:

```text
/home/dcor/vishnevsky/anaconda3/envs/DiffusionNFT
```

or the B200-compatible AirCC environment:

```text
/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/envs/nft-b200
```

The AirCC launcher uses this container:

```text
docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1
```

The CFG-distillation LoRA must contain readable copies of:

```text
adapter_config.json
adapter_model.safetensors
```

Its AirCC location is:

```text
/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/assets/cfg_distilled_sd35m_checkpoint12/lora
```

If running elsewhere, transfer that LoRA and update `BASELINE_LORA` in the
launcher. The SD3.5-medium model must also be present in the Hugging Face cache
because the configuration uses `local_files_only=true`.

W&B must be authenticated for:

```text
entity: mayavishnevsky-tel-aviv-university
project: unsupervised_creativity_nft
```

## Reference cache

The existing cache was generated with sigma minimum `0.009`. Its clean
reference latents are independent of the IEM sigma schedule and may be reused,
but its cached IEM statistics must not be used.

The launcher therefore sets:

```text
creativity.reference_cache_use_iem_statistics=false
```

and uses:

```text
/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/reference_cache/
sd35m_cfgdistilled_mixed_hpdv3pp_chat_parti_512_n20_seed0_refs256_e20_iem_s0p009_1000_g64_ws2_v1
```

with expected cache specification hash:

```text
c398ffcd6661f44127216f6dfa19ca708babdc5c547915e0675c90d0bdea7f70
```

Do not change `reference_cache_use_iem_statistics` to `true`: the run's
sigma-dependent features must be recomputed for sigma range `[10, 1000]`.

If the cache cannot be transferred, use
`config/nft.py:sd3_iem_same_prompt_mixed_no_repeat` instead of the `_cached`
configuration, remove the two `NFT_REFERENCE_CACHE_*` environment variables
and cache preflight checks, and omit both cache command-line arguments. This
regenerates references and is substantially slower but mathematically valid.

## AirCC Cycle3 submission

The current AirCC allocation is:

```text
account: cycle3_tau_patashnik_spot_prj
partition: power-gpu
qos: spot_790
GPU request: 2 x NVIDIA B200
time limit: 18 hours
```

Stage a clean immutable source snapshot. From the source machine:

```bash
SOURCE=/home/dcor/vishnevsky/unsupervised_creativity_nft
DEST=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/snapshots/source_nft_sigma_min10_20260912_v1

ssh aircc "mkdir -p '$DEST'"
rsync -az --delete \
  --exclude='.git/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='wandb/' \
  --exclude='.job_runtime/' \
  --exclude='aircc_checkpoints/' \
  --exclude='local_slurm_evaluations/' \
  --exclude='experiment_scripts/slurm_logs/' \
  "$SOURCE/" "aircc:$DEST/"
```

Submit on AirCC:

```bash
ssh aircc \
  "bash -lc 'cd /shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/snapshots/source_nft_sigma_min10_20260912_v1 && sbatch experiment_scripts/run_nft_sigma_min10_aircc_cycle3_b200_2gpu.sbatch'"
```

The already-submitted instance is AirCC job `254826`.

## Optional eight-GPU adaptation

For the strict reproduction, use two GPUs and the committed launcher. An
eight-GPU run can preserve the total 1,152 candidates per epoch and the same
effective optimization batch by changing only this geometry:

```text
NUM_GPUS: 2 -> 8
gres: gpu:nvidia_b200:2 -> gpu:nvidia_b200:8
sample.train_batch_size: 6 (unchanged per GPU)
sample.num_batches_per_epoch: 96 -> 24
train.batch_size: 8 (unchanged per GPU)
train.gradient_accumulation_steps: 72 -> 18
```

The checks are:

```text
candidate count: 6 * 8 * 24 = 1,152
effective train batch: 8 * 8 * 18 = 1,152
```

Changing the world size can alter rank-local random-number consumption and is
not guaranteed to reproduce the two-GPU sample sequence bit-for-bit. Use two
GPUs when exact stochastic comparability matters.

## Startup verification

The resolved configuration printed near the start of the log must include:

```text
beta: 0.1
train.beta: 0.25
creativity.sigma_min: 10.0
creativity.sigma_max: 1000.0
creativity.num_steps: 64
creativity.reference_cache_use_iem_statistics: false
creativity.reference_samples_per_prompt: 256
```

The launcher should also print:

```text
Reference cache: latent-only reuse; IEM statistics recomputed
```

Do not proceed if the resolved config shows `sigma_min=0.009`, cached IEM
statistics enabled, a different baseline LoRA, or a different prompt sampler.

## Resume behavior

The launcher checks:

```text
$RUN_DIR/checkpoints/latest
```

and passes `config.resume_from=$RUN_DIR` when it exists. Keep the same run
directory and W&B run ID when requeuing or resuming this exact experiment.
Do not point a different experiment at this run directory.

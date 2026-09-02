#!/bin/bash

# Append an eight-seed creative15/baseline25 panel to W&B run ababf4584bd2.
# Each prompt grid uses prompt-specific paired seeds: baseline on top, NFT below.
#SBATCH --job-name=uc-nft-199374-p8-c15
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-h200
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append

set -euo pipefail
umask 0007

ROOT=/home/dcor/vishnevsky/unsupervised_creativity_nft
RAM_ROOT=/home/dcor/vishnevsky/unsupervised_creativity_ram
SOURCE=/home/dcor/vishnevsky/immutable_sources/nft_probe_abab_c15_8seed_20260902_v2
VENV=/home/dcor/vishnevsky/anaconda3/envs/DiffusionNFT
RUN_NAME=nft_iem_same_prompt_beta0p3_199374
CHECKPOINT="$ROOT/aircc_checkpoints/$RUN_NAME/checkpoints/checkpoint-epoch-0020"
PROMPTS="$RAM_ROOT/prompts/creative_probes_legacy31_199374_190653.txt"
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
OUTPUT_DIR="$ROOT/local_slurm_evaluations/$RUN_NAME/creative_probes_epoch20_8seeds_creative15_baseline25_${SLURM_JOB_ID}"

export HF_HOME=/home/dcor/vishnevsky/.cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$RAM_ROOT/models"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-199374-p8-c15-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export WANDB_DIR="$TMPDIR/wandb-runtime"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

directories=(
  "$ROOT/experiment_scripts/slurm_logs"
  "$XDG_CACHE_HOME"
  "$WANDB_DIR/wandb"
  "$WANDB_CACHE_DIR"
  "$TRITON_CACHE_DIR"
  "$CUDA_CACHE_PATH"
  "$(dirname "$OUTPUT_DIR")"
)
mkdir -p "${directories[@]}"

required_inputs=(
  "$SOURCE/scripts/evaluate_nft_creative_probes.py"
  "$CHECKPOINT/_SUCCESS"
  "$CHECKPOINT/training_state.pt"
  "$CHECKPOINT/lora/adapter_config.json"
  "$CHECKPOINT/lora/adapter_model.safetensors"
  "$PROMPTS"
  "$BASELINE_LORA/adapter_config.json"
  "$BASELINE_LORA/adapter_model.safetensors"
  "$RAM_ROOT/models/models--stabilityai--stable-diffusion-3.5-medium/refs/main"
)
for required in "${required_inputs[@]}"; do
  [[ -r "$required" ]] || {
    echo "Missing or unreadable required input: $required" >&2
    exit 1
  }
done

[[ ! -e "$OUTPUT_DIR" ]] || {
  echo "Refusing to overwrite existing probe output: $OUTPUT_DIR" >&2
  exit 1
}

echo "Started: $(date -Is)"
echo "Checkpoint: $CHECKPOINT"
echo "Cases: 31 prompts x 8 prompt-specific seeds = 248"
echo "Schedule: NFT LoRA strength 1 for steps 1-15, then 0 for steps 16-40"
echo "W&B destination: unsupervised_creativity_nft/ababf4584bd2"
nvidia-smi

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK

cd "$SOURCE"
eval_args=(
  scripts/evaluate_nft_creative_probes.py
  --checkpoint "$CHECKPOINT"
  --prompt-file "$PROMPTS"
  --output-dir "$OUTPUT_DIR"
  --baseline-lora "$BASELINE_LORA"
  --base-seed 0
  --batch-size 2
  --seeds-per-prompt 8
  --schedule creative15_baseline25:15:25
  --wandb-project unsupervised_creativity_nft
  --wandb-entity mayavishnevsky-tel-aviv-university
  --wandb-run-id ababf4584bd2
  --wandb-name "nft_iem_same_prompt_beta0p3_199374 | creative probes | final EMA"
)
exec "$VENV/bin/python" "${eval_args[@]}"

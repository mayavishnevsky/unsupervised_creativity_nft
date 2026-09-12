#!/bin/bash

# Evaluate a copied AirCC final NFT checkpoint with the creative LoRA active
# for the first 15 of 40 steps. Submit with RUN_NAME and WANDB_RUN_ID exported.
#SBATCH --job-name=uc-nft-final-probe8
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

: "${RUN_NAME:?Submit with --export=ALL,RUN_NAME=...,WANDB_RUN_ID=...}"
: "${WANDB_RUN_ID:?Submit with --export=ALL,RUN_NAME=...,WANDB_RUN_ID=...}"

ROOT=/home/dcor/vishnevsky/unsupervised_creativity_nft
RAM_ROOT=/home/dcor/vishnevsky/unsupervised_creativity_ram
SOURCE=/home/dcor/vishnevsky/immutable_sources/nft_probe_abab_c15_8seed_20260902_v2
VENV=/home/dcor/vishnevsky/anaconda3/envs/DiffusionNFT
CHECKPOINT="$ROOT/aircc_checkpoints/$RUN_NAME/checkpoints/checkpoint-epoch-0020"
PROMPTS="$RAM_ROOT/prompts/creative_probes.txt"
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
OUTPUT_DIR="$ROOT/local_slurm_evaluations/$RUN_NAME/creative_probes_epoch20_8seeds_creative15_baseline25_${SLURM_JOB_ID}"

export HOME=/a/home/cc/cs/vishnevsky
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
export TMPDIR="/tmp/uc-nft-final-probe8-${SLURM_JOB_ID}"
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
  "$HOME/.netrc"
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
echo "Run: $RUN_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Schedule: creative 15/40 then baseline 25/40"
echo "Seeds: 8 deterministic prompt-specific seeds"
echo "W&B destination: unsupervised_creativity_nft/$WANDB_RUN_ID"
echo "Output: $OUTPUT_DIR"
nvidia-smi

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK

cd "$SOURCE"
exec "$VENV/bin/python" scripts/evaluate_nft_creative_probes.py \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --baseline-lora "$BASELINE_LORA" \
  --base-seed 0 \
  --batch-size 2 \
  --seeds-per-prompt 8 \
  --schedule creative15_baseline25:15:25 \
  --wandb-project unsupervised_creativity_nft \
  --wandb-entity mayavishnevsky-tel-aviv-university \
  --wandb-run-id "$WANDB_RUN_ID" \
  --wandb-name "$RUN_NAME | creative probes | final EMA | 8 seeds | creative15 baseline25"

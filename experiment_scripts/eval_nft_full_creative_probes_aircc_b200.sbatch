#!/bin/bash

#SBATCH --job-name=uc-nft-probes
#SBATCH --account=cycle2_tau_patashnik_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_980
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.out
#SBATCH --error=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle2_tau_patashnik_prj:/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky

set -euo pipefail
umask 0007

ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc
REPO="$ROOT/${SOURCE_DIR_NAME:?SOURCE_DIR_NAME must name the training source snapshot}"
EVAL_CODE="${EVAL_CODE_DIR:?EVAL_CODE_DIR must contain the staged evaluator}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:-196540}"
TRAIN_RUN_LABEL="${TRAIN_RUN_LABEL:-nft_iem_same_prompt_full}"
TRAIN_RUN="$ROOT/runs/${TRAIN_RUN_LABEL}_${TRAIN_JOB_ID}"
CHECKPOINT="$TRAIN_RUN/checkpoints"
PROMPTS="$EVAL_CODE/creative_probes.txt"
EVALUATIONS_DIR="$ROOT/evaluations"
OUTPUT_DIR="$EVALUATIONS_DIR/${TRAIN_RUN_LABEL}_${TRAIN_JOB_ID}_creative_probes_${SLURM_JOB_ID}"
VENV="$ROOT/envs/nft-b200"
RAM_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
ADAPTER_FULL_STRENGTH_STEPS="${ADAPTER_FULL_STRENGTH_STEPS:-}"
ADAPTER_ZERO_STRENGTH_STEPS="${ADAPTER_ZERO_STRENGTH_STEPS:-}"
ADAPTER_SCHEDULE_ARGS=()
if [[ -n "$ADAPTER_FULL_STRENGTH_STEPS" || -n "$ADAPTER_ZERO_STRENGTH_STEPS" ]]; then
  if [[ -z "$ADAPTER_FULL_STRENGTH_STEPS" || -z "$ADAPTER_ZERO_STRENGTH_STEPS" ]]; then
    echo "ADAPTER_FULL_STRENGTH_STEPS and ADAPTER_ZERO_STRENGTH_STEPS must be set together" >&2
    exit 2
  fi
  ADAPTER_SCHEDULE_ARGS=(
    --adapter-full-strength-steps "$ADAPTER_FULL_STRENGTH_STEPS"
    --adapter-zero-strength-steps "$ADAPTER_ZERO_STRENGTH_STEPS"
  )
fi

export HOME=/shared/home/users/vishnevsky
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$RAM_ROOT/cache/models"
export TRANSFORMERS_CACHE="$ROOT/cache/transformers"
export WANDB_DIR="/tmp/uc-nft-probes-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/uc-nft-probes-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-probes-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

mkdir -p "$EVALUATIONS_DIR"
chmod u+rwx "$EVALUATIONS_DIR"
for directory in "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

[[ -x "$VENV/bin/python" ]] || { echo "Missing NFT environment: $VENV" >&2; exit 2; }
[[ -f "$CHECKPOINT/latest" ]] || { echo "Missing completed training checkpoint: $CHECKPOINT/latest" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "Missing creative probes: $PROMPTS" >&2; exit 2; }
[[ -f "$EVAL_CODE/evaluate_nft_creative_probes.py" ]] || { echo "Missing evaluator: $EVAL_CODE" >&2; exit 2; }
for adapter_file in adapter_config.json adapter_model.safetensors; do
  [[ -r "$BASELINE_LORA/$adapter_file" ]] || { echo "Unreadable: $BASELINE_LORA/$adapter_file" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Training run: $TRAIN_RUN"
echo "Final checkpoint: $(<"$CHECKPOINT/latest")"
echo "Output: $OUTPUT_DIR"
nvidia-smi

cd "$REPO"
"$VENV/bin/python" "$EVAL_CODE/evaluate_nft_creative_probes.py" \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --baseline-lora "$BASELINE_LORA" \
  --base-seed 0 \
  --batch-size 2 \
  "${ADAPTER_SCHEDULE_ARGS[@]}" \
  --wandb-project unsupervised_creativity_nft \
  --wandb-entity mayavishnevsky-tel-aviv-university \
  --wandb-group "${TRAIN_RUN_LABEL}_${TRAIN_JOB_ID}" \
  --wandb-name "${TRAIN_RUN_LABEL}_${TRAIN_JOB_ID} | creative probes | final EMA"

echo "Finished: $(date -Is)"

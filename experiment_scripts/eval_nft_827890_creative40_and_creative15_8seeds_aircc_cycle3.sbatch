#!/bin/bash

# Evaluate final EMA once with eight prompt-specific seeds under two schedules.
# Each W&B image is a RAM-style grid: baseline on top, creative on bottom.
#SBATCH --job-name=uc-nft-827890-p8
#SBATCH --account=cycle3_tau_patashnik_spot_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=spot_790
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.out
#SBATCH --error=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle3_tau_patashnik_spot_prj:/shared/cycle3_tau_patashnik_spot_prj,/shared/cycle2_tau_patashnik_prj:/shared/cycle2_tau_patashnik_prj,/shared/home/users:/shared/home/users

set -euo pipefail
umask 0007

CYCLE3_ROOT=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
NFT_ROOT="$CYCLE3_ROOT/unsupervised_creativity_nft"
SOURCE="$NFT_ROOT/source_nft_probe8_dualschedule_20260902_v1"
RUN_NAME=nft_nlogp_negg_a1_1_a2_0p5_same_prompt_mixed_no_repeat_beta0p25_s0p009_1000_g64_lb4_fb16_latentcache_aircc_cycle3_resume_827890
RUN_DIR="$NFT_ROOT/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/checkpoint-epoch-0020"
PROMPTS=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/source_ram_probe_strength_20260830_v1/prompts/creative_probes.txt
BASELINE_LORA=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/assets/cfg_distilled_sd35m_checkpoint12/lora
OUTPUT_DIR="$RUN_DIR/evaluations/creative_probes_final_epoch_20_8seeds_creative40_and_creative15_baseline25_${SLURM_JOB_ID}"

export HOME=/shared/home/users/vishnevsky
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/cache/huggingface
export HF_HUB_CACHE=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/cache/models
export TRANSFORMERS_CACHE=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/cache/transformers
export WANDB_DIR="/tmp/uc-nft-827890-p8-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/uc-nft-827890-p8-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-827890-p8-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

mkdir -p "$NFT_ROOT/slurm_logs" "$OUTPUT_DIR"
for directory in "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

[[ -f "$CHECKPOINT/_SUCCESS" ]] || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CHECKPOINT/training_state.pt" ]] || { echo "Missing EMA state: $CHECKPOINT/training_state.pt" >&2; exit 2; }
[[ -f "$SOURCE/scripts/evaluate_nft_creative_probes.py" ]] || { echo "Missing evaluator" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "Missing creative probes: $PROMPTS" >&2; exit 2; }
for adapter_file in adapter_config.json adapter_model.safetensors; do
  [[ -r "$BASELINE_LORA/$adapter_file" ]] || { echo "Unreadable baseline LoRA file: $adapter_file" >&2; exit 2; }
  [[ -r "$CHECKPOINT/lora/$adapter_file" ]] || { echo "Unreadable NFT LoRA file: $adapter_file" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Checkpoint: $CHECKPOINT"
echo "Schedules: creative 40/40; creative 15/40 then baseline 25/40"
echo "Seeds: 8 deterministic, prompt-specific seeds; first seed matches the legacy probe"
echo "Baseline: SD3.5 Medium with merged CFG-distillation LoRA, guidance scale 1"
echo "Output: $OUTPUT_DIR"
nvidia-smi

cd "$SOURCE"
python scripts/evaluate_nft_creative_probes.py \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --baseline-lora "$BASELINE_LORA" \
  --base-seed 0 \
  --batch-size 2 \
  --seeds-per-prompt 8 \
  --schedule creative40:40:0 \
  --schedule creative15_baseline25:15:25 \
  --wandb-project unsupervised_creativity_nft \
  --wandb-entity mayavishnevsky-tel-aviv-university \
  --wandb-group "$RUN_NAME" \
  --wandb-name "$RUN_NAME | creative probes | final EMA | 8 seeds | creative40 and creative15 baseline25"

echo "Finished: $(date -Is)"

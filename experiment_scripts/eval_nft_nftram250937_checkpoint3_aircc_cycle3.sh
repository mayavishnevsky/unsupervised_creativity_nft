#!/bin/bash

# Evaluate checkpoint-3 EMA with eight prompt-specific seeds and full NFT LoRA.
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
SOURCE="$NFT_ROOT/snapshots/source_nft_ram_n0cz_pickclip_20260909_v3"
RUN_NAME=nft_sd35m_ram_n0czs37n_step7_strength0p8_eval_merged_pickscore_clipscore_e10_full_250937
RUN_DIR="$NFT_ROOT/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/checkpoint-epoch-0003"
PROMPTS="$SOURCE/dataset/creative_probes.txt"
OUTPUT_DIR="$RUN_DIR/evaluations/creative_probes_checkpoint3_8seeds_creative40_${SLURM_JOB_ID}"

export HOME=/shared/home/users/vishnevsky
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$NFT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$NFT_ROOT/cache/models"
export TRANSFORMERS_CACHE="$NFT_ROOT/cache/models"
export WANDB_DIR="/tmp/nft250937-ep3-probe-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/nft250937-ep3-probe-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/nft250937-ep3-probe-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/cuda"
VENV=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/envs/nft-b200

mkdir -p "$NFT_ROOT/slurm_logs" "$OUTPUT_DIR"
for directory in "$OUTPUT_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

[[ -f "$CHECKPOINT/_SUCCESS" ]] || { echo "Incomplete checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CHECKPOINT/training_state.pt" ]] || { echo "Missing EMA state: $CHECKPOINT/training_state.pt" >&2; exit 2; }
[[ -f "$SOURCE/scripts/evaluate_nft_creative_probes.py" ]] || { echo "Missing evaluator" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "Missing creative probes: $PROMPTS" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "Missing NFT environment: $VENV" >&2; exit 2; }
for adapter_file in adapter_config.json adapter_model.safetensors; do
  [[ -r "$CHECKPOINT/lora/$adapter_file" ]] || { echo "Unreadable NFT LoRA file: $adapter_file" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Checkpoint: $CHECKPOINT"
echo "Schedule: checkpoint-3 NFT LoRA active for all 40/40 steps"
echo "Seeds: 8 deterministic, prompt-specific seeds; first seed matches the legacy probe"
echo "Baseline: checkpoint config RAM evaluation adapter, merge strength 0.8, guidance scale 1"
echo "Output: $OUTPUT_DIR"
nvidia-smi

cd "$SOURCE"
"$VENV/bin/python" scripts/evaluate_nft_creative_probes.py \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --base-seed 0 \
  --batch-size 2 \
  --seeds-per-prompt 8 \
  --schedule checkpoint3_creative40:40:0 \
  --wandb-project unsupervised_creativity_nft \
  --wandb-entity mayavishnevsky-tel-aviv-university \
  --wandb-group "$RUN_NAME" \
  --wandb-run-id nftram250937 \
  --wandb-name "$RUN_NAME | creative probes | checkpoint 3 EMA | 8 seeds | creative40"

echo "Finished: $(date -Is)"

#!/bin/bash

# Evaluate checkpoint-1 EMA with eight prompt-specific seeds and full NFT LoRA.
# Each W&B image is a RAM-style grid: baseline on top, creative on bottom.
set -euo pipefail
umask 0007

CYCLE3_ROOT=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
NFT_ROOT="$CYCLE3_ROOT/unsupervised_creativity_nft"
SOURCE="$NFT_ROOT/snapshots/source_nft_ram_n0cz_pickclip_20260909_v3"
RUN_NAME=nft_sd35m_ram_n0czs37n_step7_strength0p8_eval_merged_pickscore_clipscore_e10_full_250937
RUN_DIR="$NFT_ROOT/runs/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/checkpoint-epoch-0001"
PROMPTS="$SOURCE/dataset/creative_probes.txt"
OUTPUT_DIR="$RUN_DIR/evaluations/creative_probes_checkpoint1_8seeds_creative40_${SLURM_JOB_ID}"

export HOME=/shared/home/users/vishnevsky
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$NFT_ROOT/cache/huggingface"
export HF_HUB_CACHE="$NFT_ROOT/cache/models"
export TRANSFORMERS_CACHE="$NFT_ROOT/cache/models"
export WANDB_DIR="/tmp/nft250937-ep1-probe-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/nft250937-ep1-probe-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/nft250937-ep1-probe-${SLURM_JOB_ID}"
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
[[ -f "$NFT_ROOT/launchers/evaluate_nft_creative_probes_checkpoint1.py" ]] || { echo "Missing evaluator" >&2; exit 2; }
[[ -f "$PROMPTS" ]] || { echo "Missing creative probes: $PROMPTS" >&2; exit 2; }
[[ -x "$VENV/bin/python" ]] || { echo "Missing NFT environment: $VENV" >&2; exit 2; }
for adapter_file in adapter_config.json adapter_model.safetensors; do
  [[ -r "$CHECKPOINT/lora/$adapter_file" ]] || { echo "Unreadable NFT LoRA file: $adapter_file" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Checkpoint: $CHECKPOINT"
echo "Schedule: checkpoint-1 NFT LoRA active for all 40/40 steps"
echo "Seeds: 8 deterministic, prompt-specific seeds; identical pairing to checkpoint 3"
echo "Baseline: checkpoint config RAM evaluation adapter, merge strength 0.8, guidance scale 1"
echo "Output: $OUTPUT_DIR"
nvidia-smi

cd "$SOURCE"
"$VENV/bin/python" "$NFT_ROOT/launchers/evaluate_nft_creative_probes_checkpoint1.py" \
  --checkpoint "$CHECKPOINT" \
  --prompt-file "$PROMPTS" \
  --output-dir "$OUTPUT_DIR" \
  --base-seed 0 \
  --batch-size 2 \
  --seeds-per-prompt 8 \
  --schedule checkpoint1_creative40:40:0 \
  --wandb-project unsupervised_creativity_nft \
  --wandb-entity mayavishnevsky-tel-aviv-university \
  --wandb-group "$RUN_NAME" \
  --wandb-run-id nftram250937 \
  --wandb-config-key creative_probe_checkpoint1_8seed_creative40 \
  --wandb-name "$RUN_NAME | creative probes | checkpoint 1 EMA | 8 seeds | creative40"

echo "Finished: $(date -Is)"

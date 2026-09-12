#!/bin/bash
set -euo pipefail
umask 0007

NUM_GPUS="${NFT_NUM_GPUS:-2}"
CONFIG_NAME="${NFT_CONFIG_NAME:-sd3_ram_evaluation_pickscore_clipscore_2gpu}"
SOURCE_DIR_NAME="${NFT_SOURCE_DIR_NAME:-source_nft_ram_n0cz_pickclip_20260908_v1}"
CYCLE3_ROOT=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
ROOT="$CYCLE3_ROOT/unsupervised_creativity_nft"
REPO="$ROOT/snapshots/$SOURCE_DIR_NAME"
LEGACY_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
VENV="$LEGACY_ROOT/nft_aircc/envs/nft-b200"
MODEL_CACHE="$ROOT/cache/models"
RAM_CHECKPOINT="${NFT_RAM_INPUT:-$ROOT/inputs/ram_n0czs37n_step19}"
RUN_LABEL="${NFT_RUN_LABEL:-nft_sd35m_ram_n0czs37n_eval_merged_pickscore_clipscore_e10}"
RUN_SUFFIX="${NFT_RUN_SUFFIX:-full}"
RUN_NAME="${RUN_LABEL}_${RUN_SUFFIX}_${SLURM_JOB_ID}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export HOME="$CYCLE3_ROOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$MODEL_CACHE"
export TRANSFORMERS_CACHE="$MODEL_CACHE"
# The unified cache links complete cycle-3 scorer snapshots and legacy SD3.5.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NFT_RAM_BASELINE_CHECKPOINT="$RAM_CHECKPOINT"
export NFT_RUN_NAME="$RUN_NAME"
export NFT_RUN_DIR="$RUN_DIR"
export WANDB_ENTITY=mayavishnevsky-tel-aviv-university
export WANDB_PROJECT=unsupervised_creativity_nft
export WANDB_RUN_ID="nftram${SLURM_JOB_ID}"
export WANDB_RESUME=allow
export WANDB_DIR="/tmp/uc-nft-ram-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/uc-nft-ram-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-ram-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch-inductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda"
export MPLCONFIGDIR="$TMPDIR/matplotlib"

for directory in \
  "$ROOT" "$ROOT/runs" "$ROOT/slurm_logs" "$RUN_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

required_files=(
  "$VENV/bin/python"
  "$REPO/scripts/train_nft_creativity_sd3.py"
  "$REPO/config/nft.py"
  "$REPO/dataset/pickscore/train.txt"
  "$REPO/dataset/pickscore/test.txt"
  "$REPO/dataset/creative_probes.txt"
  "$RAM_CHECKPOINT/_SUCCESS"
  "$RAM_CHECKPOINT/ram_adapters.safetensors"
)
for required in "${required_files[@]}"; do
  [[ -r "$required" ]] || { echo "Missing or unreadable input: $required" >&2; exit 2; }
done

required_models=(
  models--stabilityai--stable-diffusion-3.5-medium
  models--openai--clip-vit-large-patch14
  models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
  models--yuvalkirstain--PickScore_v1
)
for model in "${required_models[@]}"; do
  [[ -d "$MODEL_CACHE/$model" ]] || { echo "Missing cached model: $model" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Run: $RUN_NAME"
echo "Frozen source: $REPO"
echo "Base: SD3.5 Medium with RAM evaluation adapter from $RAM_CHECKPOINT merged"
echo "RAM adapter merge strength: ${NFT_RAM_MERGE_SCALE:-1.0}"
echo "Fresh NFT LoRA: rank 32, alpha 64; conditional-only sampling, CFG=1"
echo "Data: Pick-a-Pic; rewards: PickScore + CLIPScore"
echo "Validation: baseline before training, then every epoch"
echo "Final creative probes: RAM-style paired grids with 8 prompt-specific seeds"
nvidia-smi

EXPECTED_GPU_COUNT="$NUM_GPUS" "$VENV/bin/python" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} GPUs, found {torch.cuda.device_count()}")
print("torch", torch.__version__, "GPU capabilities", [
    torch.cuda.get_device_capability(index) for index in range(expected)
])
PY

RESUME_ARGS=()
if [[ -f "$RUN_DIR/checkpoints/latest" ]]; then
  RESUME_ARGS+=(--config.resume_from="$RUN_DIR")
  echo "Resuming from $(<"$RUN_DIR/checkpoints/latest")"
fi

cd "$REPO"
"$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  scripts/train_nft_creativity_sd3.py \
  --config "config/nft.py:$CONFIG_NAME" \
  "${RESUME_ARGS[@]}"

echo "Finished: $(date -Is)"

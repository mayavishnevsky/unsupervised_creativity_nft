#!/bin/bash

#SBATCH --job-name=uc-nft-iem-full
#SBATCH --account=cycle2_tau_patashnik_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_980
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.out
#SBATCH --error=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle2_tau_patashnik_prj:/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky

set -euo pipefail
umask 0007

ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc
REPO="$ROOT/${SOURCE_DIR_NAME:?SOURCE_DIR_NAME must name an immutable source snapshot}"
VENV="$ROOT/envs/nft-b200"
RAM_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
TRAIN_BETA="${TRAIN_BETA:-0.01}"
IEM_OBJECTIVE="${IEM_OBJECTIVE:-expected_squared_distance}"
DENSITY_A1="${DENSITY_A1:-1}"
DENSITY_A2="${DENSITY_A2:-1}"
IEM_NUM_STEPS="${IEM_NUM_STEPS:-64}"
IEM_LEVEL_BATCH_SIZE="${IEM_LEVEL_BATCH_SIZE:-4}"
IEM_FEATURE_BATCH_SIZE="${IEM_FEATURE_BATCH_SIZE:-2}"
EVAL_BEFORE_TRAINING="${EVAL_BEFORE_TRAINING:-true}"
EVAL_FREQ="${EVAL_FREQ:-1}"
REFERENCE_PROMPT_MODE="${REFERENCE_PROMPT_MODE:-same_prompt}"
REFERENCE_CACHE_DIR="${REFERENCE_CACHE_DIR:-}"
REFERENCE_CACHE_SPEC_SHA256="${REFERENCE_CACHE_SPEC_SHA256:-}"
REFERENCE_CACHE_NUM_INFERENCE_STEPS="${REFERENCE_CACHE_NUM_INFERENCE_STEPS:-20}"
REFERENCE_CACHE_USE_IEM_STATISTICS="${REFERENCE_CACHE_USE_IEM_STATISTICS:-true}"
CONFIG_NAME="${CONFIG_NAME:-sd3_iem_same_prompt_partiprompts}"
RUN_LABEL="${RUN_LABEL:-nft_iem_same_prompt_full}"
RUN_DIR="${RUN_DIR_OVERRIDE:-$ROOT/runs/${RUN_LABEL}_${SLURM_JOB_ID}}"
RUN_NAME="${RUN_NAME_OVERRIDE:-${RUN_LABEL}_${SLURM_JOB_ID}}"
NUM_GPUS=2
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
if ! [[ "$IEM_NUM_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "IEM_NUM_STEPS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$IEM_LEVEL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "IEM_LEVEL_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$IEM_FEATURE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "IEM_FEATURE_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if (( IEM_NUM_STEPS % IEM_LEVEL_BATCH_SIZE != 0 )); then
  echo "IEM_LEVEL_BATCH_SIZE must divide IEM_NUM_STEPS exactly" >&2
  exit 2
fi
case "$IEM_OBJECTIVE" in
  expected_squared_distance|negative_g|negative_log_p_plus_negative_g) ;;
  *)
    echo "IEM_OBJECTIVE must be expected_squared_distance, negative_g, or negative_log_p_plus_negative_g" >&2
    exit 2
    ;;
esac
for coefficient in "$DENSITY_A1" "$DENSITY_A2"; do
  if ! [[ "$coefficient" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo "DENSITY_A1 and DENSITY_A2 must be non-negative numbers" >&2
    exit 2
  fi
done
REFERENCE_CACHE_ARGS=()
if [[ -n "$REFERENCE_CACHE_DIR" || -n "$REFERENCE_CACHE_SPEC_SHA256" ]]; then
  if [[ -z "$REFERENCE_CACHE_DIR" || -z "$REFERENCE_CACHE_SPEC_SHA256" ]]; then
    echo "REFERENCE_CACHE_DIR and REFERENCE_CACHE_SPEC_SHA256 must be set together" >&2
    exit 2
  fi
  if ! [[ "$REFERENCE_CACHE_NUM_INFERENCE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "REFERENCE_CACHE_NUM_INFERENCE_STEPS must be a positive integer" >&2
    exit 2
  fi
  REFERENCE_CACHE_ARGS=(
    --config.creativity.reference_cache_dir="$REFERENCE_CACHE_DIR"
    --config.creativity.reference_cache_spec_sha256="$REFERENCE_CACHE_SPEC_SHA256"
    --config.creativity.num_inference_steps="$REFERENCE_CACHE_NUM_INFERENCE_STEPS"
  )
fi
case "$REFERENCE_CACHE_USE_IEM_STATISTICS" in
  true|false) ;;
  *)
    echo "REFERENCE_CACHE_USE_IEM_STATISTICS must be true or false" >&2
    exit 2
    ;;
esac

export HOME=/shared/home/users/vishnevsky
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$RAM_ROOT/cache/models"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export TRANSFORMERS_CACHE="$ROOT/cache/transformers"
export WANDB_DIR="/tmp/uc-nft-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/uc-nft-wandb-cache-${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch-inductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

for directory in "$ROOT" "$ROOT/envs" "$ROOT/runs" "$ROOT/slurm_logs" "$ROOT/cache" "$HF_HOME" "$TRANSFORMERS_CACHE" "$RUN_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

if [[ ! -x "$VENV/bin/python" ]]; then
  /usr/bin/python -m venv --system-site-packages "$VENV"
fi
for adapter_file in adapter_config.json adapter_model.safetensors; do
  [[ -r "$BASELINE_LORA/$adapter_file" ]] || { echo "Unreadable: $BASELINE_LORA/$adapter_file" >&2; exit 2; }
done
[[ -f "$REPO/scripts/train_nft_creativity_sd3.py" ]] || { echo "Missing source snapshot: $REPO" >&2; exit 2; }

echo "Started: $(date -Is)"
echo "Source: $REPO"
echo "Run: $RUN_DIR"
echo "KL regularization coefficient (config.train.beta): $TRAIN_BETA"
echo "IEM objective: $IEM_OBJECTIVE"
echo "Density objective coefficients: a1=$DENSITY_A1, a2=$DENSITY_A2"
echo "IEM integration intervals: $IEM_NUM_STEPS"
echo "IEM levels per denoiser forward: $IEM_LEVEL_BATCH_SIZE"
echo "IEM clean features per chunk: $IEM_FEATURE_BATCH_SIZE"
echo "Creativity reference prompt mode: $REFERENCE_PROMPT_MODE"
echo "Reference cache: ${REFERENCE_CACHE_DIR:-disabled}"
if [[ -n "$REFERENCE_CACHE_DIR" ]]; then
  echo "Reference cache spec: $REFERENCE_CACHE_SPEC_SHA256"
  echo "Cached reference inference steps: $REFERENCE_CACHE_NUM_INFERENCE_STEPS"
  echo "Use cached IEM statistics: $REFERENCE_CACHE_USE_IEM_STATISTICS"
fi
nvidia-smi

EXPECTED_GPU_COUNT="$NUM_GPUS" "$VENV/bin/python" - <<'PY'
import os
import torch
import diffusers
import peft
import transformers
import wandb
import ml_collections
import absl

expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} GPUs, found {torch.cuda.device_count()}")
capabilities = [torch.cuda.get_device_capability(i) for i in range(expected)]
if any(major < 10 for major, _ in capabilities):
    raise SystemExit(f"Expected B200 sm_100 GPUs, found {capabilities}")
print(
    "versions:",
    torch.__version__, diffusers.__version__, transformers.__version__,
    peft.__version__, wandb.__version__,
)
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
  --config "config/nft.py:${CONFIG_NAME}" \
  --config.baseline_lora_path="$BASELINE_LORA" \
  --config.creativity.reference_prompt_mode="$REFERENCE_PROMPT_MODE" \
  --config.creativity.iem_objective="$IEM_OBJECTIVE" \
  --config.creativity.density_a1="$DENSITY_A1" \
  --config.creativity.density_a2="$DENSITY_A2" \
  --config.creativity.num_steps="$IEM_NUM_STEPS" \
  --config.creativity.level_batch_size="$IEM_LEVEL_BATCH_SIZE" \
  --config.creativity.feature_batch_size="$IEM_FEATURE_BATCH_SIZE" \
  --config.creativity.reference_cache_use_iem_statistics="$REFERENCE_CACHE_USE_IEM_STATISTICS" \
  "${REFERENCE_CACHE_ARGS[@]}" \
  --config.train.beta="$TRAIN_BETA" \
  --config.eval_before_training="$EVAL_BEFORE_TRAINING" \
  --config.eval_freq="$EVAL_FREQ" \
  --config.save_dir="$RUN_DIR" \
  --config.run_name="$RUN_NAME" \
  --config.wandb_project=unsupervised_creativity_nft \
  --config.wandb_entity=mayavishnevsky-tel-aviv-university \
  "${RESUME_ARGS[@]}"

echo "Finished: $(date -Is)"

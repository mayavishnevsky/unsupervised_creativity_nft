#!/bin/bash

# Submit with DENSITY_A2 in {1,1.5,3,5,10}.
# The source snapshot is immutable so queued jobs cannot observe later edits.
#SBATCH --job-name=uc-nft-nlg
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-h200
#SBATCH --account=gpu-research
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --signal=B:TERM@120
#SBATCH --requeue

set -euo pipefail

DENSITY_A2="${DENSITY_A2:?DENSITY_A2 must be set}"
TRAIN_BETA="${TRAIN_BETA:-0.2}"
TRAIN_BETA_TAG="${TRAIN_BETA//./p}"
DENSITY_A2_TAG="${DENSITY_A2//./p}"
SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-/home/dcor/vishnevsky/immutable_sources/nft_nlogp_negg_mixed_20260819_v1}"
RUNTIME_ROOT=/home/dcor/vishnevsky/unsupervised_creativity_nft
RAM_ROOT=/home/dcor/vishnevsky/unsupervised_creativity_ram
VENV="${TRAIN_VENV:-/home/dcor/vishnevsky/anaconda3/envs/DiffusionNFT}"
EXPECTED_GPU_MAJOR="${EXPECTED_GPU_MAJOR:-9}"
HARDWARE_TAG="${HARDWARE_TAG:-local_h200}"
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
RUN_LABEL="nft_nlogp_negg_a1_1_a2_${DENSITY_A2_TAG}_same_prompt_mixed_no_repeat_beta${TRAIN_BETA_TAG}_s0p009_1000_g64_lb4_fb16_${HARDWARE_TAG}_2gpu"
RUN_DIR="$RUNTIME_ROOT/outputs/${RUN_LABEL}_${SLURM_JOB_ID}"
RUN_NAME="${RUN_LABEL}_${SLURM_JOB_ID}"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export HF_HOME=/home/dcor/vishnevsky/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export PYTHONPATH="$SOURCE_SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/uc-nft-nlg-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch-inductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

mkdir -p \
  "$RUN_DIR" \
  "$RUNTIME_ROOT/experiment_scripts/slurm_logs" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH"

for path in \
  "$SOURCE_SNAPSHOT/scripts/train_nft_creativity_sd3.py" \
  "$SOURCE_SNAPSHOT/dataset/hpdv3_plusplus/train.txt" \
  "$SOURCE_SNAPSHOT/dataset/creative_chat/creative_chat_train.txt" \
  "$SOURCE_SNAPSHOT/dataset/partiprompts_basic/train.txt" \
  "$SOURCE_SNAPSHOT/dataset/partiprompts_basic/test.txt" \
  "$SOURCE_SNAPSHOT/dataset/partiprompts_imagination/test.txt" \
  "$BASELINE_LORA/adapter_config.json" \
  "$BASELINE_LORA/adapter_model.safetensors"; do
  [[ -r "$path" ]] || { echo "Missing or unreadable required file: $path" >&2; exit 2; }
done

echo "Started: $(date -Is)"
echo "Source snapshot: $SOURCE_SNAPSHOT"
echo "Run directory: $RUN_DIR"
echo "GPUs requested: 2"
echo "Training environment: $VENV"
echo "Expected GPU compute-capability major: $EXPECTED_GPU_MAJOR"
echo "Reward: -1 * log p(x) + $DENSITY_A2 * (-g(x))"
echo "KL regularization coefficient (config.train.beta): $TRAIN_BETA"
echo "Prompt sampling: no_repeat_cycle, weights=[0.5, 0.25, 0.25]"
echo "Reference cache: disabled"
nvidia-smi

EXPECTED_GPU_COUNT=2 EXPECTED_GPU_MAJOR="$EXPECTED_GPU_MAJOR" "$VENV/bin/python" - <<'PY'
import os
from pathlib import Path
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
expected_major = int(os.environ["EXPECTED_GPU_MAJOR"])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} visible GPUs, found {torch.cuda.device_count()}")
capabilities = [torch.cuda.get_device_capability(i) for i in range(expected)]
if any(major != expected_major for major, _ in capabilities):
    raise SystemExit(
        f"Expected compute-capability major {expected_major}, found {capabilities}"
    )
if not ((Path.home() / ".netrc").is_file() or os.environ.get("WANDB_API_KEY")):
    raise SystemExit("No W&B credential found")
print(f"GPU preflight passed: capabilities={capabilities}")
PY

RESUME_ARGS=()
if [[ -f "$RUN_DIR/checkpoints/latest" ]]; then
  RESUME_ARGS+=(--config.resume_from="$RUN_DIR")
  echo "Resuming from $(<"$RUN_DIR/checkpoints/latest")"
fi

cd "$SOURCE_SNAPSHOT"
"$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  scripts/train_nft_creativity_sd3.py \
  --config config/nft.py:sd3_iem_same_prompt_mixed_no_repeat \
  --config.baseline_lora_path="$BASELINE_LORA" \
  --config.creativity.reference_prompt_mode=same_prompt \
  --config.creativity.iem_objective=negative_log_p_plus_negative_g \
  --config.creativity.density_a1=1 \
  --config.creativity.density_a2="$DENSITY_A2" \
  --config.creativity.sigma_min=0.009 \
  --config.creativity.sigma_max=1000 \
  --config.creativity.num_steps=64 \
  --config.creativity.level_batch_size=4 \
  --config.creativity.feature_batch_size=16 \
  --config.creativity.reference_cache_dir=null \
  --config.train.beta="$TRAIN_BETA" \
  --config.save_dir="$RUN_DIR" \
  --config.run_name="$RUN_NAME" \
  --config.wandb_project=unsupervised_creativity_nft \
  --config.wandb_entity=mayavishnevsky-tel-aviv-university \
  "${RESUME_ARGS[@]}"

echo "Finished: $(date -Is)"

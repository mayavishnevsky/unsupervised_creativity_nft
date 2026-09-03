#!/bin/bash

# Local-B200 copy of the evaluation-only re-noising probe for W&B d0e7a9f73f24.
# The AirCC checkpoint is mirrored locally; outputs use a distinct job-id path.
#SBATCH --job-name=uc-nft-d0e7-r5
#SBATCH --output=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.out
#SBATCH --error=/home/dcor/vishnevsky/unsupervised_creativity_nft/experiment_scripts/slurm_logs/%x_%j.err
#SBATCH --partition=gpu-b200
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=b200
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append

set -euo pipefail
umask 0007

ROOT=/home/dcor/vishnevsky/unsupervised_creativity_nft
RAM_ROOT=/home/dcor/vishnevsky/unsupervised_creativity_ram
SOURCE=/home/dcor/vishnevsky/immutable_sources/nft_probe_d0e7_renoise5_20260903_v1
VENV=/home/dcor/vishnevsky/anaconda3/envs/DiffusionNFT
RUN_NAME=nft_nlogp_negg_a1_1_a2_0p5_same_prompt_mixed_no_repeat_beta0p25_s0p009_1000_g64_lb4_fb16_latentcache_aircc_cycle3_resume_827890
CHECKPOINT="$ROOT/aircc_checkpoints/$RUN_NAME/checkpoints/checkpoint-epoch-0020"
PROMPTS="$RAM_ROOT/prompts/creative_probes.txt"
BASELINE_LORA="$RAM_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
OUTPUT_DIR="$ROOT/local_slurm_evaluations/$RUN_NAME/creative_probes_epoch20_8seeds_creative15_baseline25_renoise5_${SLURM_JOB_ID}"

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
export TMPDIR="/tmp/uc-nft-d0e7-r5-${SLURM_JOB_ID}"
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
  "$SOURCE/flow_grpo/evaluation_renoising.py"
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
echo "Cases: 43 prompts x 8 prompt-specific seeds = 344"
echo "Schedule: creative LoRA for steps 1-15, then baseline only for steps 16-40"
echo "Re-noising: 5 additional same-timestep denoising passes on each creative-active step"
echo "W&B destination: unsupervised_creativity_nft/d0e7a9f73f24"
echo "Output: $OUTPUT_DIR"
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
  --renoising-repeats 5
  --wandb-project unsupervised_creativity_nft
  --wandb-entity mayavishnevsky-tel-aviv-university
  --wandb-run-id d0e7a9f73f24
  --wandb-name "$RUN_NAME | creative probes | final EMA | creative15 baseline25 | renoise5 | 8 seeds"
)
exec "$VENV/bin/python" "${eval_args[@]}"

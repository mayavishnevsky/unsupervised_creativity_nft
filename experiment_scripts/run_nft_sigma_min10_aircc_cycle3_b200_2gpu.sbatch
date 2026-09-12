#!/bin/bash
# NFT d0e7a9f73f24 configuration with only sigma_min changed to 10. The
# existing reference cache supplies latents only; sigma-dependent IEM
# statistics are recomputed.
#SBATCH --job-name=uc-nft-s10
#SBATCH --account=cycle3_tau_patashnik_spot_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=spot_790
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --output=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.out
#SBATCH --error=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle3_tau_patashnik_spot_prj:/shared/cycle3_tau_patashnik_spot_prj,/shared/cycle2_tau_patashnik_prj:/shared/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky

set -euo pipefail
umask 0007

NUM_GPUS=2
SOURCE_DIR_NAME=source_nft_sigma_min10_20260912_v1
CYCLE3_ROOT=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
ROOT="$CYCLE3_ROOT/unsupervised_creativity_nft"
REPO="$ROOT/snapshots/$SOURCE_DIR_NAME"
LEGACY_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
VENV="$LEGACY_ROOT/nft_aircc/envs/nft-b200"
BASELINE_LORA="$LEGACY_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
CACHE_NAME=sd35m_cfgdistilled_mixed_hpdv3pp_chat_parti_512_n20_seed0_refs256_e20_iem_s0p009_1000_g64_ws2_v1
REFERENCE_CACHE="$LEGACY_ROOT/reference_cache/$CACHE_NAME"
REFERENCE_CACHE_SPEC_SHA256=c398ffcd6661f44127216f6dfa19ca708babdc5c547915e0675c90d0bdea7f70
RUN_LABEL=nft_sigma_min10_nlogp_negg_a1_1_a2_0p5_same_prompt_mixed_no_repeat_beta0p25_s10_1000_g64_lb4_fb16_latentcache_aircc_cycle3
RUN_NAME="${RUN_LABEL}_${SLURM_JOB_ID}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export HOME="$CYCLE3_ROOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$LEGACY_ROOT/cache/huggingface"
export HF_HUB_CACHE="$LEGACY_ROOT/cache/models"
export TRANSFORMERS_CACHE="$LEGACY_ROOT/nft_aircc/cache/transformers"
export NFT_REFERENCE_CACHE_DIR="$REFERENCE_CACHE"
export NFT_REFERENCE_CACHE_SPEC_SHA256="$REFERENCE_CACHE_SPEC_SHA256"
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb-cache"
export WANDB_RUN_ID="nfts10${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-s10-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCH_EXTENSIONS_DIR="$TMPDIR/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch-inductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda"

for directory in \
  "$ROOT" \
  "$ROOT/runs" \
  "$ROOT/slurm_logs" \
  "$RUN_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH"; do
  mkdir -p "$directory"
  chmod u+rwx "$directory"
done

for required in \
  "$VENV/bin/python" \
  "$REPO/scripts/train_nft_creativity_sd3.py" \
  "$REPO/dataset/hpdv3_plusplus/train.txt" \
  "$REPO/dataset/creative_chat/creative_chat_train.txt" \
  "$REPO/dataset/partiprompts_basic/train.txt" \
  "$REPO/dataset/partiprompts_basic/test.txt" \
  "$REPO/dataset/partiprompts_imagination/test.txt" \
  "$BASELINE_LORA/adapter_config.json" \
  "$BASELINE_LORA/adapter_model.safetensors" \
  "$REFERENCE_CACHE/_SUCCESS" \
  "$REFERENCE_CACHE/spec.json"; do
  [[ -r "$required" ]] || { echo "Missing or unreadable input: $required" >&2; exit 2; }
done
CACHE_SPEC_SHA256=$("$VENV/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["spec_sha256"])' "$REFERENCE_CACHE/spec.json")
test "$CACHE_SPEC_SHA256" = "$REFERENCE_CACHE_SPEC_SHA256"

echo "Started: $(date -Is)"
echo "Run: $RUN_NAME"
echo "Frozen source: $REPO"
echo "Source configuration: NFT W&B d0e7a9f73f24"
echo "Reward: -log p(x) + 0.5 * (-g(x))"
echo "IEM schedule: sigma=[10, 1000], 64 integration levels"
echo "KL regularization coefficient: 0.25"
echo "Reference cache: latent-only reuse; IEM statistics recomputed"
nvidia-smi

EXPECTED_GPU_COUNT="$NUM_GPUS" "$VENV/bin/python" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(f"Expected {expected} GPUs, found {torch.cuda.device_count()}")
capabilities = [torch.cuda.get_device_capability(i) for i in range(expected)]
if any(major < 10 for major, _ in capabilities):
    raise SystemExit(f"Expected B200 sm_100 GPUs, found {capabilities}")
print("torch", torch.__version__, "GPU capabilities", capabilities)
PY

RESUME_ARGS=()
if [[ -f "$RUN_DIR/checkpoints/latest" ]]; then
  RESUME_ARGS+=(--config.resume_from="$RUN_DIR")
  echo "Resuming from $(<"$RUN_DIR/checkpoints/latest")"
fi

cd "$REPO"
exec "$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  scripts/train_nft_creativity_sd3.py \
  --config config/nft.py:sd3_iem_same_prompt_mixed_no_repeat_cached \
  --config.baseline_lora_path="$BASELINE_LORA" \
  --config.num_epochs=20 \
  --config.resolution=512 \
  --config.seed=0 \
  --config.mixed_precision=bf16 \
  --config.eval_before_training=false \
  --config.eval_freq=1 \
  --config.save_freq=1 \
  --config.sample.num_steps=25 \
  --config.sample.eval_num_steps=40 \
  --config.sample.guidance_scale=1.0 \
  --config.sample.num_image_per_prompt=24 \
  --config.sample.num_prompt_groups=48 \
  --config.sample.train_batch_size=6 \
  --config.sample.num_batches_per_epoch=96 \
  --config.sample.test_batch_size=2 \
  --config.train.beta=0.25 \
  --config.train.batch_size=8 \
  --config.train.gradient_accumulation_steps=72 \
  --config.train.learning_rate=0.0003 \
  --config.train.adam_beta1=0.9 \
  --config.train.adam_beta2=0.99 \
  --config.train.adam_weight_decay=0.0001 \
  --config.train.lora_rank=32 \
  --config.train.lora_alpha=64 \
  --config.creativity.iem_objective=negative_log_p_plus_negative_g \
  --config.creativity.density_a1=1.0 \
  --config.creativity.density_a2=0.5 \
  --config.creativity.sigma_min=10 \
  --config.creativity.sigma_max=1000 \
  --config.creativity.num_steps=64 \
  --config.creativity.level_batch_size=4 \
  --config.creativity.feature_batch_size=16 \
  --config.creativity.reference_prompt_mode=same_prompt \
  --config.creativity.reference_samples_per_prompt=256 \
  --config.creativity.reference_selection_mode=all \
  --config.creativity.reference_cache_use_iem_statistics=false \
  --config.creativity.num_inference_steps=20 \
  --config.run_name="$RUN_NAME" \
  --config.save_dir="$RUN_DIR" \
  --config.wandb_project=unsupervised_creativity_nft \
  --config.wandb_entity=mayavishnevsky-tel-aviv-university \
  "${RESUME_ARGS[@]}"

#!/bin/bash
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

: "${SOURCE_DIR_NAME:?SOURCE_DIR_NAME must name an immutable source snapshot}"
: "${DIVERSITY_METHOD:?DIVERSITY_METHOD must be cads or contextual_repulsion}"

NUM_GPUS=2
CYCLE3_ROOT=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
ROOT="$CYCLE3_ROOT/unsupervised_creativity_nft"
REPO="$ROOT/snapshots/$SOURCE_DIR_NAME"
LEGACY_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
VENV="$LEGACY_ROOT/nft_aircc/envs/nft-b200"
BASELINE_LORA="$LEGACY_ROOT/assets/cfg_distilled_sd35m_checkpoint12/lora"
CACHE_NAME=sd35m_cfgdistilled_mixed_hpdv3pp_chat_parti_512_n20_seed0_refs256_e20_iem_s0p009_1000_g64_ws2_v1
REFERENCE_CACHE="$LEGACY_ROOT/reference_cache/$CACHE_NAME"
REFERENCE_CACHE_SPEC_SHA256=c398ffcd6661f44127216f6dfa19ca708babdc5c547915e0675c90d0bdea7f70

case "$DIVERSITY_METHOD" in
  cads)
    DIVERSITY_LABEL=cads_t0p80_0p98_s0p20_psi0p5
    SAMPLE_BATCH_SIZE=6
    SAMPLE_BATCHES_PER_EPOCH=96
    DIVERSITY_ARGS=(
      --config.candidate_diversity.method=cads
      --config.candidate_diversity.cads.tau1=0.80
      --config.candidate_diversity.cads.tau2=0.98
      --config.candidate_diversity.cads.noise_scale=0.20
      --config.candidate_diversity.cads.psi=0.5
      --config.candidate_diversity.cads.rescale=true
    )
    ;;
  contextual_repulsion)
    DIVERSITY_LABEL=contextual_text_tstop6_n50_eta1e8
    # Contextual repulsion jointly transforms one complete same-prompt group.
    SAMPLE_BATCH_SIZE=24
    SAMPLE_BATCHES_PER_EPOCH=24
    DIVERSITY_ARGS=(
      --config.candidate_diversity.method=contextual_repulsion
      --config.candidate_diversity.contextual_repulsion.target_tensor=text
      --config.candidate_diversity.contextual_repulsion.t_stop=6
      --config.candidate_diversity.contextual_repulsion.num_steps=50
      --config.candidate_diversity.contextual_repulsion.eta=100000000
      --config.candidate_diversity.contextual_repulsion.kernel_reg=0.0001
    )
    ;;
  *)
    echo "DIVERSITY_METHOD must be cads or contextual_repulsion; got $DIVERSITY_METHOD" >&2
    exit 2
    ;;
esac

RUN_LABEL="nft_nlogp_negg_a1_1_a2_0p5_same_prompt_mixed_no_repeat_beta0p25_s0p009_1000_g64_lb4_fb16_${DIVERSITY_LABEL}_latentcache_aircc_cycle3"
RUN_NAME="${RUN_LABEL}_${SLURM_JOB_ID}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

export HOME="$CYCLE3_ROOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$LEGACY_ROOT/cache/huggingface"
export HF_HUB_CACHE="$LEGACY_ROOT/cache/models"
export TRANSFORMERS_CACHE="$LEGACY_ROOT/nft_aircc/cache/transformers"
export WANDB_DIR="/tmp/uc-nft-diversity-wandb-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="/tmp/uc-nft-diversity-wandb-cache-${SLURM_JOB_ID}"
export WANDB_RUN_ID="nftdiv${SLURM_JOB_ID}"
export WANDB_SERVICE_WAIT_SECONDS=300
export WANDB__SERVICE_WAIT=300
export TMPDIR="/tmp/uc-nft-diversity-${SLURM_JOB_ID}"
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

test -x "$VENV/bin/python"
test -f "$REPO/scripts/train_nft_creativity_sd3.py"
test -f "$REPO/flow_grpo/candidate_diversity.py"
test -f "$BASELINE_LORA/adapter_config.json"
test -f "$BASELINE_LORA/adapter_model.safetensors"
test -f "$REFERENCE_CACHE/_SUCCESS"
test -f "$REFERENCE_CACHE/spec.json"
CACHE_SPEC_SHA256=$("$VENV/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["spec_sha256"])' "$REFERENCE_CACHE/spec.json")
test "$CACHE_SPEC_SHA256" = "$REFERENCE_CACHE_SPEC_SHA256"

echo "Started: $(date -Is)"
echo "Source: $REPO"
echo "Run: $RUN_DIR"
echo "Candidate diversity: $DIVERSITY_LABEL (training candidates only)"
echo "Sampling geometry: 48 prompts x 24 candidates; local batch=$SAMPLE_BATCH_SIZE"
echo "References, validation, and optimization conditioning remain unmodified."
echo "Reference cache: $REFERENCE_CACHE"
echo "Reference cache spec: $REFERENCE_CACHE_SPEC_SHA256"
echo "Cached IEM statistics: enabled with schedule and per-noise hash checks."
nvidia-smi

EXPECTED_GPU_COUNT="$NUM_GPUS" "$VENV/bin/python" - <<'PY'
import os
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
if torch.cuda.device_count() != expected:
    raise SystemExit(
        f"Expected {expected} GPUs, found {torch.cuda.device_count()}"
    )
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
"$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  scripts/train_nft_creativity_sd3.py \
  --config config/nft.py:sd3_iem_same_prompt_mixed_no_repeat \
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
  --config.sample.train_batch_size="$SAMPLE_BATCH_SIZE" \
  --config.sample.num_batches_per_epoch="$SAMPLE_BATCHES_PER_EPOCH" \
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
  --config.creativity.sigma_min=0.009 \
  --config.creativity.sigma_max=1000 \
  --config.creativity.num_steps=64 \
  --config.creativity.level_batch_size=4 \
  --config.creativity.feature_batch_size=16 \
  --config.creativity.reference_prompt_mode=same_prompt \
  --config.creativity.reference_samples_per_prompt=256 \
  --config.creativity.reference_selection_mode=all \
  --config.creativity.reference_cache_dir="$REFERENCE_CACHE" \
  --config.creativity.reference_cache_spec_sha256="$REFERENCE_CACHE_SPEC_SHA256" \
  --config.creativity.reference_cache_use_iem_statistics=true \
  --config.creativity.num_inference_steps=20 \
  --config.run_name="$RUN_NAME" \
  --config.save_dir="$RUN_DIR" \
  --config.wandb_project=unsupervised_creativity_nft \
  --config.wandb_entity=mayavishnevsky-tel-aviv-university \
  "${DIVERSITY_ARGS[@]}" \
  "${RESUME_ARGS[@]}"

echo "Finished: $(date -Is)"

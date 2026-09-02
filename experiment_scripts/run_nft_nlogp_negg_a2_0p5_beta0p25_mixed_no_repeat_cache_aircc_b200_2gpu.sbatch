#!/bin/bash

# NFT counterpart of RAM run 221234. Reuse its generated reference latents,
# but recompute IEM features because the cached mu/v use an older convention.
#SBATCH --job-name=uc-nft-a2p5-b25-cache
#SBATCH --account=cycle2_tau_patashnik_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_980
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --output=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.out
#SBATCH --error=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle2_tau_patashnik_prj:/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky

set -euo pipefail

RAM_ROOT=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc
CACHE_NAME=sd35m_cfgdistilled_mixed_hpdv3pp_chat_parti_512_n20_seed0_refs256_e20_iem_s0p009_1000_g64_ws2_v1

export SOURCE_DIR_NAME=source_nft_nlogp_negg_a2p5_beta0p25_cache_20260829_v1
export CONFIG_NAME=sd3_iem_same_prompt_mixed_no_repeat
export RUN_LABEL=nft_nlogp_negg_a1_1_a2_0p5_same_prompt_mixed_no_repeat_beta0p25_s0p009_1000_g64_lb4_fb16_latentcache
export TRAIN_BETA=0.25
export IEM_OBJECTIVE=negative_log_p_plus_negative_g
export DENSITY_A1=1
export DENSITY_A2=0.5
export IEM_NUM_STEPS=64
export IEM_LEVEL_BATCH_SIZE=4
export IEM_FEATURE_BATCH_SIZE=16
export REFERENCE_PROMPT_MODE=same_prompt
export REFERENCE_CACHE_DIR="$RAM_ROOT/reference_cache/$CACHE_NAME"
export REFERENCE_CACHE_SPEC_SHA256=c398ffcd6661f44127216f6dfa19ca708babdc5c547915e0675c90d0bdea7f70
export REFERENCE_CACHE_NUM_INFERENCE_STEPS=20
export REFERENCE_CACHE_USE_IEM_STATISTICS=false
export EVAL_BEFORE_TRAINING=false
export EVAL_FREQ=1

exec bash "$RAM_ROOT/nft_aircc/${SOURCE_DIR_NAME}/experiment_scripts/run_nft_iem_same_prompt_full_aircc_b200_2gpu.sbatch"

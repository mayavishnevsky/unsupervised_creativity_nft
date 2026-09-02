#!/bin/bash

# Sweep wrapper for -log p(x) + a2 * (-g(x)) on the fixed mixed prompt cycle.
# Submit with DENSITY_A2 in {1,1.5,3,5,10}.
#SBATCH --job-name=uc-nft-nlogp-negg
#SBATCH --account=cycle2_tau_patashnik_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_980
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --output=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.out
#SBATCH --error=/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/slurm_logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --container-image="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
#SBATCH --container-mounts=/shared/cycle2_tau_patashnik_prj:/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky

set -euo pipefail

export SOURCE_DIR_NAME="${SOURCE_DIR_NAME:-source_nlogp_negg_mixed_20260819_v1}"
export DENSITY_A2="${DENSITY_A2:?DENSITY_A2 must be set}"
DENSITY_A2_TAG="${DENSITY_A2//./p}"
export CONFIG_NAME=sd3_iem_same_prompt_mixed_no_repeat
export RUN_LABEL="nft_nlogp_negg_a1_1_a2_${DENSITY_A2_TAG}_same_prompt_mixed_no_repeat_beta0p2_s0p009_1000_g64_lb4_fb16"
export TRAIN_BETA=0.2
export IEM_OBJECTIVE=negative_log_p_plus_negative_g
export DENSITY_A1=1
export IEM_NUM_STEPS=64
export IEM_LEVEL_BATCH_SIZE=4
export IEM_FEATURE_BATCH_SIZE=16
export REFERENCE_PROMPT_MODE=same_prompt

exec bash "/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/${SOURCE_DIR_NAME}/experiment_scripts/run_nft_iem_same_prompt_full_aircc_b200_2gpu.sbatch"

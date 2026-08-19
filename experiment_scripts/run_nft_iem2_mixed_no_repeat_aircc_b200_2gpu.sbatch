#!/bin/bash

#SBATCH --job-name=uc-nft-iem2-mix-nr
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

export SOURCE_DIR_NAME=source_no_repeat_mixed_iem2_20260819_v2
export CONFIG_NAME=sd3_iem_same_prompt_mixed_no_repeat
export RUN_LABEL=nft_iem2_same_prompt_mixed_no_repeat_beta0p2_s0p009_1000_g64_lb4_fb16
export TRAIN_BETA=0.2
export IEM_OBJECTIVE=expected_squared_distance
export IEM_NUM_STEPS=64
export IEM_LEVEL_BATCH_SIZE=4
export IEM_FEATURE_BATCH_SIZE=16
export REFERENCE_PROMPT_MODE=same_prompt

exec bash "/shared/home/users/vishnevsky/unsupervised_creativity_ram_aircc/nft_aircc/${SOURCE_DIR_NAME}/experiment_scripts/run_nft_iem_same_prompt_full_aircc_b200_2gpu.sbatch"

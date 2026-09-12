#!/bin/bash
#SBATCH --job-name=nft-ram-pickclip
#SBATCH --account=cycle3_tau_patashnik_spot_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=spot_790
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_b200:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --output=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.out
#SBATCH --error=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky/unsupervised_creativity_nft/slurm_logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail
umask 0077

CYCLE3_HOME=/shared/cycle3_tau_patashnik_spot_prj/users/vishnevsky
NEBIUS_AUTH_HOME=/shared/home/users/vishnevsky
ROOT="$CYCLE3_HOME/unsupervised_creativity_nft"
INNER_LAUNCHER="$ROOT/launchers/run_nft_ram_n0czs37n_pickclip_aircc_cycle3_b200.sh"
PRIVATE_IMAGE="docker://cr.me-west1.nebius.cloud#i00bj4w9jsrfcfe6s2/paretonft-train:v1"
CONTAINER_MOUNTS="/shared/cycle3_tau_patashnik_spot_prj:/shared/cycle3_tau_patashnik_spot_prj,/shared/cycle2_tau_patashnik_prj:/shared/cycle2_tau_patashnik_prj,/shared/home/users/vishnevsky:/shared/home/users/vishnevsky"
NUM_GPUS="${NFT_NUM_GPUS:-2}"

export HOME="$CYCLE3_HOME"
TOKEN_JSON="${TMPDIR:-/tmp}/nebius-registry-${SLURM_JOB_ID}.json"
CREDENTIALS_DIR="$HOME/.config/enroot"
CREDENTIALS_FILE="$CREDENTIALS_DIR/.credentials"
CREDENTIALS_TMP="$CREDENTIALS_DIR/.credentials.${SLURM_JOB_ID}.tmp"

mkdir -p "$CREDENTIALS_DIR"
command -v nebius >/dev/null
HOME="$NEBIUS_AUTH_HOME" nebius iam whoami >/dev/null
printf '%s\n' cr.me-west1.nebius.cloud \
  | HOME="$NEBIUS_AUTH_HOME" nebius registry docker-credential get --format json > "$TOKEN_JSON"

python3 - "$TOKEN_JSON" "$CREDENTIALS_TMP" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    credential = json.load(handle)
server = credential.get("ServerURL") or credential.get("server_url")
username = credential.get("Username") or credential.get("username")
secret = credential.get("Secret") or credential.get("secret")
if not all((server, username, secret)):
    raise SystemExit(f"Unexpected docker credential fields: {sorted(credential)}")
server = server.removeprefix("https://").removeprefix("http://").rstrip("/")
with open(destination, "w", encoding="utf-8") as handle:
    handle.write(f"machine {server} login {username} password {secret}\n")
os.chmod(destination, 0o600)
PY
mv -f "$CREDENTIALS_TMP" "$CREDENTIALS_FILE"
rm -f "$TOKEN_JSON"

[[ -r "$INNER_LAUNCHER" ]] || {
  echo "Missing inner launcher: $INNER_LAUNCHER" >&2
  exit 2
}

srun --ntasks=1 \
  --gpus-per-task="$NUM_GPUS" \
  --container-image="$PRIVATE_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  /bin/bash "$INNER_LAUNCHER"

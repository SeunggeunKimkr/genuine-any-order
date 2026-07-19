#!/bin/bash
# =============================================================================
# Pre-tokenize the training corpora into fixed-width .bin shards under
# $PRETOKENIZED_ROOT. Uses the dataset list + filters in the config.
# After this, run scripts/merge_manifests.py and (optionally)
# scripts/drop_truncated_rows.py. See docs/DATA.md.
#
# SLURM account/partition below are PLACEHOLDERS. See docs/CLUSTER.md.
# =============================================================================
#SBATCH --job-name=flexmdm_pretok
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Optional env: LIMIT, DATASETS, PRETOKENIZE_BATCH_SIZE, LOG_EVERY, OVERWRITE, QUIET.

set -eo pipefail

# Under sbatch, SLURM copies this script to a spool dir, so BASH_SOURCE does
# not point into the repo; prefer REPO_ROOT / the sbatch submission dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
TOKENIZERS_PARALLELISM=true
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/flexmdm/config/fsdp_train.yaml}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}/flexmdm_tokenized}"
PRETOKENIZED_ROOT="${PRETOKENIZED_ROOT:-${SCRATCH_ROOT}/pretokenized}"
LOG_EVERY="${LOG_EVERY:-10000}"
LIMIT="${LIMIT:-}"
DATASETS="${DATASETS:-}"
PRETOKENIZE_BATCH_SIZE="${PRETOKENIZE_BATCH_SIZE:-}"
OVERWRITE="${OVERWRITE:-0}"
QUIET="${QUIET:-0}"

set -u
mkdir -p "$SCRATCH_ROOT" "$PRETOKENIZED_ROOT"
cd "$REPO_ROOT"

cmd=(python -u -m flexmdm.data tokenize --config "$CONFIG_PATH"
     --output-root "$PRETOKENIZED_ROOT" --log-every "$LOG_EVERY")
[ -n "$LIMIT" ] && cmd+=(--limit "$LIMIT")
[ -n "$PRETOKENIZE_BATCH_SIZE" ] && cmd+=(--batch-size "$PRETOKENIZE_BATCH_SIZE")
if [ -n "$DATASETS" ]; then
  # shellcheck disable=SC2206
  dataset_args=($DATASETS); cmd+=(--datasets "${dataset_args[@]}")
fi
[ "$OVERWRITE" = "1" ] && cmd+=(--overwrite)
[ "$QUIET" = "1" ] && cmd+=(--quiet)

echo "=== FlexMDM pre-tokenization ==="
echo "config=$CONFIG_PATH  pretokenized_root=$PRETOKENIZED_ROOT"
echo "datasets=${DATASETS:-config_default}  limit=${LIMIT:-full}  overwrite=$OVERWRITE"
printf 'cmd:'; printf ' %q' "${cmd[@]}"; printf '\n'

srun --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 "${cmd[@]}"

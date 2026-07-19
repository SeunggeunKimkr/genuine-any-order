#!/bin/bash
#SBATCH --job-name=anyorder
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=0-06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Compute any-order metrics (CBC, RUB, OBW per evals/metrics.md) for traces
# produced by run.sh. Pure CPU — the metric computation is AST + token
# alignment.
#
# Launch:
#   OUTPUT_ROOT=/path/to/run TOKENIZER=/path/to/Dream-Coder-v0-Base-7B \
#     sbatch evals/run_anyorder.sh

set -eo pipefail

# Under sbatch, SLURM copies this script to a spool dir; see run.sh.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT (the dir generate.py wrote into)}"
TOKENIZER="${TOKENIZER:-${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}}"

DATASETS="${DATASETS:-humaneval mbpp}"  # any-order metrics use generation-level datasets
# Per-model decoding algs, matching what generate.py produced (published:
# FlexMDM top_k, Dream-Coder entropy).
MODEL_ALGS="${MODEL_ALGS:-flexmdm=top_k dreamcoder=entropy}"
N_SAMPLES="${N_SAMPLES:-16}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
INCLUDE_REFERENCE_NODES="${INCLUDE_REFERENCE_NODES:-false}"  # code-only by default
LIMIT="${LIMIT:-}"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

set -u
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

echo "=== any-order metrics ==="
echo "output_root=$OUTPUT_ROOT"
echo "datasets=$DATASETS  model_algs=$MODEL_ALGS  include_ref=$INCLUDE_REFERENCE_NODES"

CMD=(python -m evals.tree_analysis.cli
     --output-root "$OUTPUT_ROOT"
     --datasets $DATASETS
     --model-algs $MODEL_ALGS
     --n-samples "$N_SAMPLES"
     --workers "$WORKERS"
     --tokenizer "$TOKENIZER")

if [ "$INCLUDE_REFERENCE_NODES" = "true" ]; then
  CMD+=(--include-reference-nodes)
else
  CMD+=(--no-include-reference-nodes)
fi

if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo "=== any-order done ==="

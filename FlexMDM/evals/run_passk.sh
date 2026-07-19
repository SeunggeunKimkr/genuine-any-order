#!/bin/bash
#SBATCH --job-name=passk
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=0-06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Compute pass@k for traces produced by run.sh.
# Reads <OUTPUT_ROOT>/<gen_dataset>/raw/...; writes <OUTPUT_ROOT>/passk/...
#
# Launch:
#   OUTPUT_ROOT=/path/to/run TOKENIZER=/path/to/Dream-Coder-v0-Base-7B \
#     sbatch evals/run_passk.sh
#
# Note: this is CPU-only — pass@k just runs sandboxed Python subprocs.

set -eo pipefail

# Under sbatch, SLURM copies this script to a spool dir; see run.sh.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT (the dir generate.py wrote into)}"
TOKENIZER="${TOKENIZER:-${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}}"

DATASETS="${DATASETS:-humaneval humaneval_plus mbpp mbpp_plus}"
# Per-model decoding algs, matching what generate.py produced (published:
# FlexMDM top_k, Dream-Coder entropy). Scoring an alg that was never
# generated is a hard error in passk.py.
MODEL_ALGS="${MODEL_ALGS:-flexmdm=top_k dreamcoder=entropy}"
MODES="${MODES:-prompt_tail_sanitize prompt_cleaned_sanitize prompt_tail_raw cleaned_raw}"
N_SAMPLES="${N_SAMPLES:-16}"
KS="${KS:-1 2 4 8 16}"
TIMEOUT="${TIMEOUT:-30.0}"   # 30 s = the published scoring timeout
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
LIMIT="${LIMIT:-}"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

set -u
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

echo "=== pass@k ==="
echo "output_root=$OUTPUT_ROOT"
echo "datasets=$DATASETS  model_algs=$MODEL_ALGS"
echo "modes=$MODES  ks=$KS  timeout=$TIMEOUT  workers=$WORKERS"

CMD=(python -m evals.humaneval_compare.passk
     --output-root "$OUTPUT_ROOT"
     --datasets $DATASETS
     --model-algs $MODEL_ALGS
     --modes $MODES
     --n-samples "$N_SAMPLES"
     --ks $KS
     --timeout "$TIMEOUT"
     --workers "$WORKERS"
     --tokenizer "$TOKENIZER")

if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

# Extraction-robust (any-of-4 extraction modes) pass@k — the headline
# convention in the released tables. Writes $OUTPUT_ROOT/passk/robust_summary.json.
ROBUST_CMD=(python -m evals.humaneval_compare.robust_passk
             --output-root "$OUTPUT_ROOT"
             --ks $KS
             --n-samples "$N_SAMPLES")
echo "+ ${ROBUST_CMD[*]}"
"${ROBUST_CMD[@]}"

echo "=== pass@k done (summary.json + robust_summary.json) ==="

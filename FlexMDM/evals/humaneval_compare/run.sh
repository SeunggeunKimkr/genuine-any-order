#!/bin/bash
#SBATCH --job-name=heval_cmp
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=0-08:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Generate code-eval traces for one (MODEL, DATASET) pair, sharded across an
# array job. HE/HE+ share generation under DATASET=humaneval; MBPP/MBPP+ share
# generation under DATASET=mbpp. Pass@k and any-order metrics consume the .pt
# traces this script produces.
#
# Launch examples:
#   MODEL=flexmdm    DATASET=humaneval sbatch --array=0-3  evals/humaneval_compare/run.sh
#   MODEL=flexmdm    DATASET=mbpp      sbatch --array=0-15 evals/humaneval_compare/run.sh
#   MODEL=dreamcoder DATASET=humaneval sbatch --array=0-3  evals/humaneval_compare/run.sh
#   MODEL=dreamcoder DATASET=mbpp      sbatch --array=0-15 evals/humaneval_compare/run.sh
#
# Each shard is one H100. Work units = (task, alg). For HE: 164 × |algs| units;
# for MBPP: 378 × |algs| units. Defaults: NUM_SHARDS=4 (HE) / 16 (MBPP) —
# MBPP at max length 1100 with eager attention is by far the slowest run;
# 16 shards keep it comfortably inside the 8 h walltime.
# The sbatch --array range must be exactly 0..NUM_SHARDS-1 (fewer array
# tasks than NUM_SHARDS silently leaves that shard's units ungenerated —
# the pass@k preflight will catch it at scoring time).
# Each unit generates N_SAMPLES=16 in batched calls; interrupted runs can be
# resubmitted as-is (--skip-existing resumes; seeds are content-addressed).

set -eo pipefail

# Under sbatch, SLURM copies this script to a spool dir, so BASH_SOURCE does
# not point into the repo. Resolution order: explicit REPO_ROOT, then the
# sbatch submission dir (the documented usage submits from the repo root),
# then BASH_SOURCE (direct bash invocation).
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVAL_OUTPUT_ROOT:-./eval_runs}/default}"

MODEL="${MODEL:?set MODEL=flexmdm or MODEL=dreamcoder}"
DATASET="${DATASET:?set DATASET=humaneval|humaneval_plus|mbpp|mbpp_plus}"
FLEX_CKPT="${FLEX_CKPT:-${SAVE_ROOT:-./checkpoints}/global_step_49500}"
DREAM_MODEL="${DREAM_MODEL:-${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}}"

# Generation hyperparameters. Per-model defaults reproduce the PUBLISHED runs
# with no overrides:
#   flexmdm    -> top_k confidence, temperature 0.1, power schedules with the
#                 inference-time reparameterization to a-bar=2.9 (train a=1.7,
#                 unmask a*b=2.89), Poisson insertion, eager attention (the
#                 bit-exact setting — see evals/REPRODUCIBILITY.md).
#   dreamcoder -> negative-entropy confidence at Dream-Coder's official
#                 temperatures (HE 0.2 / MBPP 0.1), sdpa attention. NOTE: the
#                 paper's Dream-Coder pass@k rows instead used
#                 TEMPERATURE=1.0 N_SAMPLES=16 (for pass@k diversity); set
#                 those explicitly to reproduce them. The defaults here are
#                 the any-order-metrics configuration.
# Every value can still be overridden via the environment.
STEPS="${STEPS:-512}"
TOP_P="${TOP_P:-0.9}"
N_SAMPLES="${N_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INSERTION_COUNT_SAMPLER="${INSERTION_COUNT_SAMPLER:-poisson}"
LIMIT="${LIMIT:-}"          # set to N for sanity runs; empty = full set
STRATIFIED="${STRATIFIED:-}"  # set to "1" + LIMIT to take stratified subset

# Published max lengths: HumanEval 768, MBPP 1100 (total sequence cap).
case "$DATASET" in
  mbpp|mbpp_plus) MAX_LENGTH="${MAX_LENGTH:-1100}" ;;
  *)              MAX_LENGTH="${MAX_LENGTH:-768}" ;;
esac

if [ "$MODEL" = "flexmdm" ]; then
  ALGS="${ALGS:-top_k}"
  ATTN_IMPL="${ATTN_IMPL:-eager}"
  TEMPERATURE="${TEMPERATURE:-0.1}"
  INSERTION_SCHEDULE="${INSERTION_SCHEDULE:-power}"
  UNMASKING_SCHEDULE="${UNMASKING_SCHEDULE:-power}"
  INSERTION_EXPONENT="${INSERTION_EXPONENT:-2.9}"
  UNMASKING_EXPONENT="${UNMASKING_EXPONENT:-2.89}"
  TRAIN_INSERTION_SCHEDULE="${TRAIN_INSERTION_SCHEDULE:-power}"
  TRAIN_INSERTION_EXPONENT="${TRAIN_INSERTION_EXPONENT:-1.7}"
else
  ALGS="${ALGS:-entropy}"
  ATTN_IMPL="${ATTN_IMPL:-sdpa}"
  INSERTION_SCHEDULE="${INSERTION_SCHEDULE:-linear}"
  UNMASKING_SCHEDULE="${UNMASKING_SCHEDULE:-quadratic}"
  INSERTION_EXPONENT="${INSERTION_EXPONENT:-}"
  UNMASKING_EXPONENT="${UNMASKING_EXPONENT:-}"
  TRAIN_INSERTION_SCHEDULE="${TRAIN_INSERTION_SCHEDULE:-}"
  TRAIN_INSERTION_EXPONENT="${TRAIN_INSERTION_EXPONENT:-}"
  # Dream-Coder official temperatures: HE 0.2, MBPP 0.1.
  case "$DATASET" in
    mbpp|mbpp_plus) TEMPERATURE="${TEMPERATURE:-0.1}" ;;
    *)              TEMPERATURE="${TEMPERATURE:-0.2}" ;;
  esac
fi

# Default shard counts (see header). Must match the sbatch --array range.
case "$DATASET" in
  mbpp|mbpp_plus) NUM_SHARDS="${NUM_SHARDS:-16}" ;;
  *)              NUM_SHARDS="${NUM_SHARDS:-4}" ;;
esac
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

set -u

mkdir -p "$OUTPUT_ROOT" "$REPO_ROOT/evals/humaneval_compare/outputs"

export TOKENIZERS_PARALLELISM=false
# Set HF_HUB_OFFLINE=1 (e.g. in paths.env) on air-gapped compute nodes AFTER
# warming the HF cache; the default (0) lets a fresh checkout download the
# task datasets and models on first use.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$REPO_ROOT"

echo "=== code-eval generation ==="
echo "host=$(hostname)"
echo "model=$MODEL  dataset=$DATASET"
echo "output_root=$OUTPUT_ROOT"
echo "shard=$SHARD_INDEX / $NUM_SHARDS"
echo "algs=$ALGS  steps=$STEPS  temp=$TEMPERATURE  top_p=$TOP_P  attn=$ATTN_IMPL"
echo "max_length=$MAX_LENGTH  n_samples=$N_SAMPLES  limit=${LIMIT:-all}"
echo "schedules: insertion=$INSERTION_SCHEDULE($INSERTION_EXPONENT) unmasking=$UNMASKING_SCHEDULE($UNMASKING_EXPONENT) train_insertion=$TRAIN_INSERTION_SCHEDULE($TRAIN_INSERTION_EXPONENT)"
echo "insertion_count_sampler=$INSERTION_COUNT_SAMPLER"

CMD=(python -m evals.humaneval_compare.generate
     --model "$MODEL"
     --dataset "$DATASET"
     --output-root "$OUTPUT_ROOT"
     --shard-index "$SHARD_INDEX"
     --num-shards "$NUM_SHARDS"
     --n-samples "$N_SAMPLES"
     --batch-size "$BATCH_SIZE"
     --algs $ALGS
     --steps "$STEPS"
     --temperature "$TEMPERATURE"
     --top-p "$TOP_P"
     --max-length "$MAX_LENGTH"
     --insertion-schedule "$INSERTION_SCHEDULE"
     --unmasking-schedule "$UNMASKING_SCHEDULE"
     --insertion-count-sampler "$INSERTION_COUNT_SAMPLER"
     --attn-implementation "$ATTN_IMPL")

if [ "$MODEL" = "flexmdm" ]; then
  CMD+=(--checkpoint-dir "$FLEX_CKPT")
else
  CMD+=(--dreamcoder-model "$DREAM_MODEL")
fi

# Optional schedule-reparameterization args (only appended when set).
[ -n "$INSERTION_EXPONENT" ]        && CMD+=(--insertion-exponent "$INSERTION_EXPONENT")
[ -n "$UNMASKING_EXPONENT" ]        && CMD+=(--unmasking-exponent "$UNMASKING_EXPONENT")
[ -n "$TRAIN_INSERTION_SCHEDULE" ]  && CMD+=(--train-insertion-schedule "$TRAIN_INSERTION_SCHEDULE")
[ -n "$TRAIN_INSERTION_EXPONENT" ]  && CMD+=(--train-insertion-exponent "$TRAIN_INSERTION_EXPONENT")

if [ -n "$LIMIT" ]; then
  CMD+=(--limit "$LIMIT")
  if [ -n "$STRATIFIED" ]; then
    CMD+=(--stratified)
  fi
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"

echo "=== done shard $SHARD_INDEX ==="

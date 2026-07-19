#!/bin/bash
# Top-level driver for the full eval pipeline. Runs:
#   (1) Generation: (FlexMDM, DC-base) × (HumanEval, MBPP), 16 samples each.
#       Per-model defaults in run.sh are the published configurations
#       (FlexMDM: top_k / temp 0.1 / a-bar=2.9 power reparam / eager;
#        DC-base: entropy at HE 0.2 / MBPP 0.1).
#   (2) Pass@k (per-extraction-mode + extraction-robust any-of-4),
#       k ∈ {1,2,4,8,16}, timeout 30 s — reproduces the paper's FlexMDM tables.
#   (3) Any-order metrics: CBC/RUB/OBW (HE + MBPP).
#
# NOTE on the Dream-Coder baseline: the paper's DC pass@k rows used
# TEMPERATURE=1.0 N_SAMPLES=16 (pass@k diversity setting). This driver
# generates DC at its any-order-metrics defaults instead; to reproduce the
# DC pass@k rows, submit a separate generation run:
#   MODEL=dreamcoder DATASET=humaneval TEMPERATURE=1.0 N_SAMPLES=16 \
#     sbatch --array=0-3 evals/humaneval_compare/run.sh
#
# This is a *driver*, not an sbatch — it submits sbatch jobs and prints the
# job dependencies so the user can monitor them.
#
# Usage:
#   OUTPUT_ROOT=/path/to/run \
#   FLEX_CKPT=/path/to/global_step_49500 \
#     bash evals/run_full_eval.sh
#
# Optional sanity-mode (only first 4 tasks per dataset, fewer samples):
#   SANITY=1 OUTPUT_ROOT=/tmp/sanity FLEX_CKPT=... bash evals/run_full_eval.sh

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
FLEX_CKPT="${FLEX_CKPT:?set FLEX_CKPT}"
TOKENIZER="${TOKENIZER:-${BASE_MODEL:-Dream-org/Dream-Coder-v0-Base-7B}}"

if [ "${SANITY:-}" = "1" ]; then
    LIMIT="${LIMIT:-4}"
    N_SAMPLES="${N_SAMPLES:-4}"
    NUM_SHARDS_HE="${NUM_SHARDS_HE:-1}"
    NUM_SHARDS_MBPP="${NUM_SHARDS_MBPP:-1}"
    STEPS="${STEPS:-128}"
else
    LIMIT="${LIMIT:-}"
    N_SAMPLES="${N_SAMPLES:-16}"
    NUM_SHARDS_HE="${NUM_SHARDS_HE:-4}"
    # MBPP (max len 1100, eager) is the slowest run; 16 shards keep headroom.
    NUM_SHARDS_MBPP="${NUM_SHARDS_MBPP:-16}"
    STEPS="${STEPS:-512}"
fi

mkdir -p "$OUTPUT_ROOT"

echo "=== plan ==="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "FLEX_CKPT=$FLEX_CKPT"
echo "SANITY=${SANITY:-0}  LIMIT=${LIMIT:-all}  N_SAMPLES=$N_SAMPLES  STEPS=$STEPS"
echo "NUM_SHARDS: HE=$NUM_SHARDS_HE  MBPP=$NUM_SHARDS_MBPP"

submit_gen () {
    local model=$1
    local dataset=$2
    local num_shards=$3
    local label="gen_${model}_${dataset}"

    local last=$(( num_shards - 1 ))
    local jid
    jid=$(REPO_ROOT=$REPO_ROOT \
          MODEL=$model DATASET=$dataset \
          OUTPUT_ROOT=$OUTPUT_ROOT \
          FLEX_CKPT=$FLEX_CKPT \
          NUM_SHARDS=$num_shards \
          N_SAMPLES=$N_SAMPLES \
          STEPS=$STEPS \
          LIMIT=$LIMIT \
          sbatch --parsable --array=0-$last \
                 --job-name=$label \
                 "$REPO_ROOT/evals/humaneval_compare/run.sh")
    if [ -z "$jid" ]; then
        echo "sbatch submission failed for $label" >&2
        exit 1
    fi
    # sbatch --parsable for an array prints "<jobid>"; depend on the whole array via afterok:<jobid>
    echo "$jid"
}

# (1) Generation
gen_flex_he=$(submit_gen flexmdm humaneval $NUM_SHARDS_HE)
echo "submitted gen_flexmdm_humaneval: $gen_flex_he"
gen_flex_mbpp=$(submit_gen flexmdm mbpp $NUM_SHARDS_MBPP)
echo "submitted gen_flexmdm_mbpp: $gen_flex_mbpp"
gen_dc_he=$(submit_gen dreamcoder humaneval $NUM_SHARDS_HE)
echo "submitted gen_dreamcoder_humaneval: $gen_dc_he"
gen_dc_mbpp=$(submit_gen dreamcoder mbpp $NUM_SHARDS_MBPP)
echo "submitted gen_dreamcoder_mbpp: $gen_dc_mbpp"

GEN_DEP="afterok:$gen_flex_he:$gen_flex_mbpp:$gen_dc_he:$gen_dc_mbpp"

# (2) Pass@k
passk_jid=$(REPO_ROOT=$REPO_ROOT \
            OUTPUT_ROOT=$OUTPUT_ROOT TOKENIZER=$TOKENIZER \
            N_SAMPLES=$N_SAMPLES \
            LIMIT=$LIMIT \
            sbatch --parsable --dependency=$GEN_DEP \
                   --job-name=passk \
                   "$REPO_ROOT/evals/run_passk.sh")
echo "submitted passk: $passk_jid (depends on generation)"

# (3) Any-order
anyorder_jid=$(REPO_ROOT=$REPO_ROOT \
               OUTPUT_ROOT=$OUTPUT_ROOT TOKENIZER=$TOKENIZER \
               N_SAMPLES=$N_SAMPLES \
               LIMIT=$LIMIT \
               sbatch --parsable --dependency=$GEN_DEP \
                      --job-name=anyorder \
                      "$REPO_ROOT/evals/run_anyorder.sh")
echo "submitted anyorder: $anyorder_jid (depends on generation)"

cat <<EOF

=== submitted ===
gen_flexmdm_humaneval=$gen_flex_he
gen_flexmdm_mbpp=$gen_flex_mbpp
gen_dreamcoder_humaneval=$gen_dc_he
gen_dreamcoder_mbpp=$gen_dc_mbpp
passk=$passk_jid (after gen)
anyorder=$anyorder_jid (after gen)

monitor:
  squeue -u \$USER -o '%.10i %.10P %.20j %.8T %.10M %R'

results:
  passk (strict):  $OUTPUT_ROOT/passk/summary.json
  passk (robust):  $OUTPUT_ROOT/passk/robust_summary.json   <- headline tables
  anyorder:        $OUTPUT_ROOT/anyorder/summary.json
EOF

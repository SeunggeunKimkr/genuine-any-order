# Bit-exact reproducibility of the pass@k evaluation

The reported pass@k numbers are computed from 16 samples per task, generated
by a single deterministic procedure. Re-running the command below in the
pinned environment reproduces the full sample set bit-for-bit.

## Determinism design

For every (task, alg) unit, generation runs in chunks of `--batch-size 8`
samples. Before each chunk the global torch RNG is reseeded as

    base_seed = int.from_bytes(sha256(f"{task_id}|{model}|{alg}")[:4], "big")
    torch.manual_seed(base_seed + chunk_start)      # chunk_start ∈ {0, 8}

All stochastic ops (Gumbel top-k unmasking, Poisson insertion counts) draw
exclusively from the torch RNG — there is no `random`/`numpy` randomness in
the sampling path — so sample k is a pure function of
(task_id, alg, chunk, code, environment, GPU architecture). The seeds are
content-addressed: they are fixed by the sha256 formula before any result
exists, and extending n cannot change existing samples — sample k is
identical no matter which `--n-samples ≥ k+1` run produced it. This also
means a run can be resumed or extended in place (`--skip-existing`, on by
default, skips already-generated chunks and produces exactly what a fresh
single run would).

## Pinned environment

Bit-exactness holds within a fixed numerical stack — NVIDIA H100, bfloat16,
eager attention — and the pinned package set (`evals/requirements-repro.txt`):

    python 3.10 · torch 2.5.1+cu124 · transformers 4.46.2
    tokenizers 0.20.3 · numpy 2.2.6

(On a different GPU architecture or torch version the samples remain draws
from the same distribution, but bit-level equality is not guaranteed.)

The scoring side additionally needs the **execution deps** listed in
`evals/requirements-repro.txt` (pytest, scipy, matplotlib, IPython, flask,
beautifulsoup4, selenium): generated completions may import them at test
time, and a module missing from the scoring environment turns an otherwise
passing sample into an import failure. This mostly affects the Dream-Coder
baseline (temperature 1.0 generations import liberally); FlexMDM samples do
not import beyond the core set.

## Command (one run per dataset)

HumanEval (paste-safe; no inline comments):

    python -m evals.humaneval_compare.generate \
        --model flexmdm --checkpoint-dir <ckpt> \
        --dataset humaneval \
        --output-root <out> \
        --n-samples 16 --batch-size 8 --algs top_k \
        --steps 512 --temperature 0.1 --top-p 0.9 \
        --max-length 768 \
        --insertion-schedule power --unmasking-schedule power \
        --insertion-exponent 2.9 --unmasking-exponent 2.89 \
        --train-insertion-schedule power --train-insertion-exponent 1.7 \
        --insertion-count-sampler poisson \
        --torch-dtype bfloat16 --attn-implementation eager

MBPP: identical except `--dataset mbpp --max-length 1100`.

(Shard with `--shard-index/--num-shards` freely: sharding assigns whole units
to jobs and cannot affect any sample's value. `evals/humaneval_compare/run.sh`
with `MODEL=flexmdm` defaults to exactly this configuration, including the
per-dataset max length — so `MODEL=flexmdm DATASET=humaneval sbatch --array=0-3
evals/humaneval_compare/run.sh` needs no overrides.)

Scoring (unbiased Chen et al. pass@k estimator over all 16 samples, plus the
extraction-robust any-of-4-modes aggregation; 30 s is the published scoring
timeout and the code default):

    python -m evals.humaneval_compare.passk --output-root <out> \
        --datasets humaneval humaneval_plus --model-algs flexmdm=top_k \
        --n-samples 16 --ks 1 2 4 8 16 --timeout 30.0 --tokenizer <ckpt>
    python -m evals.humaneval_compare.robust_passk --output-root <out> \
        --ks 1 2 4 8 16 --n-samples 16

`evals/run_passk.sh` runs both, writing `passk/summary.json` (per-extraction-
mode) and `passk/robust_summary.json` (the headline tables).

Generation is bit-exact; scoring is not quite — test execution uses a
wall-clock `--timeout` (published setting and code default: **30 s**), so a
sample whose test suite runs near the cap can flip with CPU speed. Measured
behavior on identical traces:

- **@ 30 s** (the published setting): machine-robust — across two machines,
  every entry of every variant is identical except a ≤0.02 pt wobble on MBPP+
  pass@1 from a single boundary task (`Mbpp/599`, suite runtime ≈30 s; it
  passes whenever it finishes, so the higher reading is the timeout→∞ value).
- **base variants**: timeout-insensitive — every base-row timeout is a
  non-terminating program, so the exact value of the cap does not affect
  them.
- **plus variants**: need the full 30 s — with a shorter cap, borderline-slow
  suites time out, and the plus rows drop and become CPU-speed-sensitive.

So score with the 30 s default and compare against the tables in
`docs/REPRODUCE.md`.

## Verification

`scripts/compare_traces.py` compares two output roots record by record (full
512-step trajectory tensors, not just final sequences). We verified that
regenerating evaluation chunks in the pinned environment on a different H100
node months later reproduces the originally generated traces bit-for-bit
(96/96 records across HumanEval and MBPP spot checks).

Task lists are persisted with the traces (`<output_root>/<dataset>/tasks.json`);
digests of the sets used for the paper:

    sha256(humaneval/tasks.json) = 30a9b4f74b86620991b5deef66e3cfd99f4e271a56a9a13f154a7e059a7ba96b
    sha256(mbpp/tasks.json)      = 609bf338ed51ee017c91e8f37880496b0cf479114d7e9cfffb899dd04e5b8a55

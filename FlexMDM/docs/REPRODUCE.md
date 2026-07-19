# Reproducing FlexMDM

End-to-end: set up the environment, rebuild the tokenized data, train (or
download the checkpoint), then generate and score.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate flexmdm
pip install -r requirements.txt
pip install -e .        # makes `flexmdm` and `evals` importable; Hydra finds flexmdm/config
```

The environment pins Python 3.10 and PyTorch 2.5.1 / CUDA 12.4 (matching the
training run); adjust `pytorch-cuda` in `environment.yml` for your driver/toolkit.
Optional Flash-Attention: `pip install --no-build-isolation flash-attn==2.8.3`.

Copy `paths.env.template` to `paths.env` and edit it for your machine (env, base
model, caches, data/checkpoint/eval roots, W&B). See [`CLUSTER.md`](CLUSTER.md).

## 2. Build the tokenized dataset

Pre-tokenize → drop-truncated → merge. Full instructions in [`DATA.md`](DATA.md):

```bash
python -m flexmdm.data tokenize --config flexmdm/config/fsdp_train.yaml --output-root "$PRETOKENIZED_ROOT"
python scripts/drop_truncated_rows.py --root "$PRETOKENIZED_ROOT"
python scripts/merge_manifests.py    --root "$PRETOKENIZED_ROOT"
```

## 3. Train (or download the checkpoint)

**Train** on 16× H100 (4 nodes × 4 GPU), FSDP HYBRID_SHARD, ~3 days:

```bash
sbatch scripts/train_16gpu.sh
```

This launches `flexmdm.train_fsdp` with `--config-name fsdp_train` (the
released training recipe). Set SLURM account/partition and paths first — see
[`CLUSTER.md`](CLUSTER.md). The released/evaluated checkpoint is
`global_step_49500`.

**Or download** the released inference weights from the Hugging Face Hub instead
of training:

```bash
hf download yuyuanchen0/flexmdm --local-dir "$FLEX_CKPT"
```

(https://huggingface.co/yuyuanchen0/flexmdm — the directory is
self-contained; point `FLEX_CKPT` at it for the evaluation commands below.)

## 4. Generate and score

The driver submits generation (FlexMDM + Dream-Coder baseline × HumanEval + MBPP),
then pass@k, then any-order metrics:

```bash
OUTPUT_ROOT=/path/to/run \
FLEX_CKPT=/path/to/checkpoints/global_step_49500 \
  bash evals/run_full_eval.sh
```

Smoke test (few tasks/samples):

```bash
SANITY=1 OUTPUT_ROOT=/tmp/sanity FLEX_CKPT=... bash evals/run_full_eval.sh
```

Results land at `$OUTPUT_ROOT/passk/summary.json` and
`$OUTPUT_ROOT/anyorder/summary.json`.

### FlexMDM generation configuration (the published setting)

The released numbers use **top-k confidence unmasking** (`--algs top_k`) with
the inference-time schedule reparameterization to **ā = 2.9** from the training
schedule **a = 1.7** (the model is queried at `τ(t) = 1 − (1 − t)^(2.9/1.7)`)
at **temperature 0.1**, with **eager attention** (the bit-exact setting). As
direct `evals/humaneval_compare/generate.py` flags:

```
--algs top_k \
--insertion-schedule power --insertion-exponent 2.9 \
--train-insertion-schedule power --train-insertion-exponent 1.7 \
--unmasking-schedule power --unmasking-exponent 2.89 \
--insertion-count-sampler poisson \
--steps 512 --top-p 0.9 --temperature 0.1 \
--max-length 768 \
--torch-dtype bfloat16 --attn-implementation eager
```

with `--max-length 768` for HumanEval and `--max-length 1100` for MBPP.

**These are exactly the `MODEL=flexmdm` defaults of
`evals/humaneval_compare/run.sh`** (including the per-dataset max length), so
no overrides are needed:

```bash
MODEL=flexmdm DATASET=humaneval sbatch --array=0-3  evals/humaneval_compare/run.sh
MODEL=flexmdm DATASET=mbpp      sbatch --array=0-15 evals/humaneval_compare/run.sh
```

Every value can still be overridden via the environment (see the header of
`run.sh`), and the environment propagates through `evals/run_full_eval.sh`.

**Dream-Coder baseline settings.** The paper's Dream-Coder **pass@k** rows
used `ALGS=entropy TEMPERATURE=1.0 N_SAMPLES=16` (higher temperature for
pass@k diversity; Appendix D.3). The `MODEL=dreamcoder` defaults of `run.sh`
are instead Dream-Coder's official decoding temperatures (HumanEval 0.2 /
MBPP 0.1, entropy) — the configuration used for the **any-order metrics**. To
reproduce the DC pass@k rows, submit a separate generation with those
variables set explicitly.

## Expected numbers

FlexMDM `global_step_49500`, pass@k (extraction-robust any-of-4 grading) from
n = 16 samples/task, scored with the 30 s test timeout — exactly what the
pipeline's `MODEL=flexmdm` defaults produce end-to-end, and exactly the
paper's Table 5 FlexMDM rows. Generation is bit-reproducible per
[`../evals/REPRODUCIBILITY.md`](../evals/REPRODUCIBILITY.md):

| Benchmark | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|---|
| HumanEval  | 50.65 | 66.60 | 78.69 | 86.86 | 92.07 |
| HumanEval+ | 46.61 | 61.89 | 73.83 | 82.07 | 87.80 |
| MBPP       | 62.22 | 74.61 | 81.81 | 86.19 | 89.68 |
| MBPP+      | 52.86 | 64.38 | 72.04 | 77.00 | 80.16 |

### Timeout sensitivity

Scoring executes each sample against the benchmark's full test suite under a
wall-clock `--timeout`; **30 s is the published setting and the code
default**. Measured behavior on identical traces:

- **At 30 s, scoring is machine-robust.** Across two machines, every
  per-sample result is identical except one boundary task (`Mbpp/599`, whose
  MBPP+ suite itself runs ≈30 s), worth ≤0.02 pt on MBPP+ pass@1 (52.86 vs
  52.84). The suite passes whenever it finishes, so the higher reading is the
  timeout→∞ value; the table above uses it.
- **Base rows are timeout-insensitive**: every base-row timeout is a
  non-terminating program, so the exact value of the cap does not affect
  them.
- **Plus rows need the full 30 s.** The plus variants run thousands of test
  cases per problem; with a shorter cap, borderline-slow suites time out, and
  the plus rows drop and become CPU-speed-sensitive (up to ~1 pt). Do not
  lower `--timeout` if you want to match the published tables.

Dream-Coder-7B-Base baseline rows (`ALGS=entropy TEMPERATURE=1.0
N_SAMPLES=16`, scored identically — 30 s timeout, extraction-robust):

| Benchmark | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|---|
| HumanEval  | 58.65 | 72.22 | 81.34 | 86.94 | 90.85 |
| HumanEval+ | 53.89 | 67.05 | 75.96 | 81.53 | 85.37 |
| MBPP       | 64.53 | 77.33 | 84.66 | 89.29 | 92.86 |
| MBPP+      | 53.98 | 66.13 | 73.55 | 78.25 | 82.01 |

Verified on two machines: every Dream-Coder entry is machine-exact except
MBPP+, where the same `Mbpp/599` boundary suite wobbles ≤0.04 pt at low k
(it is the *suite* that runs ≈30 s, regardless of which model wrote the
solution); as above, the MBPP+ row uses the timeout→∞ values. Scoring
Dream-Coder additionally requires the execution deps in
`evals/requirements-repro.txt` — its temperature-1.0 completions import
liberally, and a missing module turns a passing sample into an import
failure (up to −1.9 pt on these rows in a minimal environment).

### Any-order metrics

The tree-based any-order metrics — **CBC** (Coverage Before Commitment), **RUB**
(Return to Unfinished Blocks), **RUB+** (graded RUB), and **OBW** (Open-Block
Width) — are computed by `evals/tree_analysis` (see `evals/metrics.md`). FlexMDM
scores **substantially higher** than Dream-Coder, reflecting genuine any-order
traversal. Paper Table 4 (`code_only`, overall averages):

| Benchmark | Model | CBC | RUB | RUB+ | OBW |
|---|---|---|---|---|---|
| HumanEval | Dream-Coder | 0.742 | 0.596 | 0.583 | 0.745 |
| HumanEval | **FlexMDM** | **0.835** | **0.780** | **0.720** | **0.857** |
| MBPP | Dream-Coder | 0.798 | 0.666 | 0.664 | 0.798 |
| MBPP | **FlexMDM** | **0.876** | **0.823** | **0.768** | **0.891** |

## Notes on variance

- **pass@k** is reported with the **extraction-robust "any-of-4"** convention:
  a sample counts as correct if any of the four code-extraction modes
  (`prompt_tail_sanitize`, `prompt_cleaned_sanitize`, `prompt_tail_raw`,
  `cleaned_raw`) yields a passing program — the score is invariant to
  extraction-convention quirks and is applied identically to every model.
  `evals/run_passk.sh` computes it automatically and writes
  `$OUTPUT_ROOT/passk/robust_summary.json` (keyed per `dataset|model|alg`);
  the per-mode tables land next to it in `passk/summary.json`.
- In the **pinned environment** (H100, eager attention — the script defaults)
  generation is deterministic, so there is **no** run-to-run variation: the
  table above reproduces bit-for-bit (see
  [`../evals/REPRODUCIBILITY.md`](../evals/REPRODUCIBILITY.md)). On a different
  GPU architecture / torch version the Poisson-insertion stochasticity makes
  results vary slightly run-to-run, most visibly at **pass@1**.
- Seeds are content-addressed per sample chunk, so resuming or extending a run
  (`--skip-existing`, a larger `--n-samples`) never changes existing samples.
- Scoring uses a **30 s** whole-suite wall-clock timeout (the published
  setting and the `passk.py` default) — see the timeout-sensitivity section
  above for the measured behavior at other timeouts.

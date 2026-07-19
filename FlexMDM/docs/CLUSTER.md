# Cluster & Environment Setup

The launch scripts target a **SLURM** cluster (originally a Harvard FASRC-style
setup). Nothing cluster-specific is hardcoded in the scripts that you cannot
override; this page lists what you must set for your own site. All examples use
placeholders — substitute your own accounts, partitions, and paths.

## 1. `paths.env`

`scripts/env.sh` is sourced by every job script and auto-loads a `paths.env` from
the repo root if present. Copy the template and edit it:

```bash
cp paths.env.template paths.env
$EDITOR paths.env
```

Variables (from `paths.env.template`):

| Variable | Purpose |
|---|---|
| `CONDA_ENV` | conda env name (default `flexmdm`) |
| `CONDA_PREFIX_OVERRIDE` | absolute env prefix; used instead of `CONDA_ENV` if set |
| `CUDA_MODULE` | optional Lmod module, e.g. `cuda/12.4.1-fasrc01` |
| `BASE_MODEL` | HF id or local path to Dream-Coder-v0-Base-7B (also the tokenizer) |
| `TOKENIZER` | tokenizer path for the eval scripts (usually same as `BASE_MODEL`) |
| `HF_HOME` | Hugging Face cache root |
| `SCRATCH_ROOT` | where `flexmdm.data tokenize` writes raw corpora |
| `PRETOKENIZED_ROOT` | root of the tokenized `.bin` shards (default `$SCRATCH_ROOT/pretokenized`) |
| `SAVE_ROOT` | training checkpoint output dir |
| `EVAL_OUTPUT_ROOT` | evaluation output root |
| `WANDB_ENTITY` / `WANDB_PROJECT` / `WANDB_MODE` / `WANDB_DIR` | Weights & Biases |
| `MASTER_ADDR_SUFFIX` | FQDN suffix for the master node (FASRC: `.rc.fas.harvard.edu`) |

The training config (`flexmdm/config/fsdp_train.yaml`) reads several of these via
`${oc.env:VAR,default}` (e.g. `BASE_MODEL`, `SCRATCH_ROOT`, `PRETOKENIZED_ROOT`,
`SAVE_ROOT`, `WANDB_*`).

Eval-side wiring: the driver `evals/run_full_eval.sh` requires `OUTPUT_ROOT` (the
specific run directory) and `FLEX_CKPT`; `EVAL_OUTPUT_ROOT` from `paths.env` is
used as the parent default by `evals/humaneval_compare/run.sh`, `FLEX_CKPT` falls
back to `$SAVE_ROOT/global_step_49500`, and `TOKENIZER` falls back to
`BASE_MODEL`.

## 2. SLURM account and partition

Every `sbatch` script ships with **placeholder** SLURM directives you must edit
(or override at submit time):

```
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
```

Choose partitions appropriate to each job's resources:

| Script | Resources |
|---|---|
| `scripts/train_16gpu.sh` | 4 nodes × 4 GPU (H100-class), ~3 days |
| `scripts/pretokenize.sh` | 1 node, 1 GPU, ~1 day |
| `evals/humaneval_compare/run.sh` (generation) | 1 GPU per array shard |
| `evals/run_passk.sh`, `evals/run_anyorder.sh` | CPU-only (no GPU) |

Override at submit time if you prefer:
`sbatch --account=<your-account> --partition=<your-partition> scripts/train_16gpu.sh`.

These scripts were originally run on a Harvard FASRC (Kempner) cluster; no
site-specific account or partition is committed in the release.

## 3. Conda + CUDA

`scripts/env.sh` activates conda and (optionally) loads a CUDA module:

- Set `CONDA_ENV` (name on `PATH`) or `CONDA_PREFIX_OVERRIDE` (absolute prefix).
- `CONDA_BASE` is auto-detected from `CONDA_EXE` / `conda info --base` if unset.
- If your cluster uses Lmod, set `CUDA_MODULE` (e.g. `cuda/12.4.1-fasrc01`).
- `HF_HOME` defaults to `~/.cache/huggingface`.

> Site note: on the original cluster, a broken `libffi.so.8` symlink in the env
> required prepending the system libffi to `LD_LIBRARY_PATH`. This workaround is
> intentionally **not** hardcoded in `scripts/env.sh`; add it in your `paths.env`
> if you hit the same issue.

## 4. Multi-node training (torchrun)

`scripts/train_16gpu.sh` runs **4 nodes × 4 GPUs = 16 GPUs**, FSDP
`hybrid_shard` (shard within a node, replicate across nodes). Under `srun` it
launches:

```bash
python -m torch.distributed.run \
  --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
  --node_rank "$SLURM_NODEID" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  -m flexmdm.train_fsdp --config-name "$CONFIG_NAME" ...
```

Cluster-specific pieces:

- `NNODES` / `NPROC_PER_NODE` default to 4 / 4 (override for a different topology).
- `MASTER_ADDR` = first node from `scontrol show hostnames` + `MASTER_ADDR_SUFFIX`
  (set the suffix to your cluster's FQDN, e.g. `.rc.fas.harvard.edu`).
- `MASTER_PORT` is derived from `SLURM_JOB_ID` (override `MASTER_PORT` if needed).
- **NCCL interface:** `NCCL_SOCKET_IFNAME` defaults to the auto-detected default
  route interface. On an InfiniBand fabric, set it explicitly, e.g.
  `NCCL_SOCKET_IFNAME=ib0` (and `GLOO_SOCKET_IFNAME` similarly).
- `CONFIG_NAME` defaults to `fsdp_train` (the released training recipe). Pass extra
  Hydra overrides via `EXTRA_OVERRIDES="key=value ..."`; set `AUTO_RESUME=1` to
  resume from the latest complete checkpoint under `$SAVE_ROOT`.

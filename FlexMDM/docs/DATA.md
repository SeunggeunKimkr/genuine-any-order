# Training Data

The released checkpoint was trained on a mixture of **five
Hugging Face sources**, filtered and tokenized with the **Dream-Coder tokenizer**
to a fixed length of **768**, then up-sampled by integer repeat factors. This is
the "pretokenized" mixture, and it **includes rStar-Coder and KodCode-V1-SFT-4o**.

All dataset specifications, filters, and repeat factors below are implemented in
`flexmdm/data.py` and configured in `flexmdm/config/fsdp_train.yaml`.

## Sources

| Alias | HF dataset | Config | Prompt → Answer | Filter | Repeat | License |
|---|---|---|---|---|---|---|
| `rstarcoder-seed-sft` | `microsoft/rStar-Coder` | `seed_sft` | `question` (+ `starter_code`) → `code` | `is_passed` required | ×2 | CC BY 4.0 |
| `rstarcoder-synthetic-sft` | `microsoft/rStar-Coder` | `synthetic_sft` | `question` → `code` | `is_passed` required **if the field is present** | ×2 | CC BY 4.0 |
| `opc-sft-stage2-educational` | `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | `instruction` → `code` | none | ×2 | MIT |
| `KodCode-V1-SFT-4o` | `KodCode/KodCode-V1-SFT-4o` | — | `question` → `4o_solution` (fallback `solution`) | none | ×2 | **CC BY-NC 4.0** (non-commercial) |
| `OpenCodeInstruct-score1-py-all` | `nvidia/OpenCodeInstruct` | — | `input` → `output` | `average_test_score >= 0.9`; Python-only extraction; strip trailing demo / `__main__` / print blocks | ×1 | CC BY 4.0 |

Notes:

- Filters, fields, and repeat factors are verified against `flexmdm/data.py`
  (`DATASET_SPECS`) and the `dataloader_repeat_factors` block in
  `flexmdm/config/fsdp_train.yaml`. In the code the default repeat factor is 2 for
  every source except `OpenCodeInstruct-score1-py-all`, which is 1.
- `rStar-Coder` license (`cc-by-4.0`) confirmed from the dataset's Hugging Face
  card. The other three are as listed on their respective HF pages. Licenses can
  change — re-check each source's dataset card at release time.
- BOS is prepended and the prompt/answer boundary uses an internal-pad separator
  (`prepend_bos: true`, `separator_kind: internal_pad`).

## What we release (and what we do not)

We release the **tokenization scripts and the dataset manifest**, **not** the
tokenized data arrays.

**You must review and accept each source's license/terms** before downloading or
using the data.

## Reconstructing the tokenized dataset

Prerequisites: a working environment and a `paths.env` (see
[`CLUSTER.md`](CLUSTER.md)) that sets `SCRATCH_ROOT` and `PRETOKENIZED_ROOT`, and
`BASE_MODEL` (the Dream-Coder tokenizer). Run from the repository root.

### 1. Pre-tokenize the sources

```bash
python -m flexmdm.data tokenize \
  --config flexmdm/config/fsdp_train.yaml \
  --output-root "$PRETOKENIZED_ROOT"
```

Or, as a SLURM job (same command, cluster wrapper):

```bash
sbatch scripts/pretokenize.sh
```

This writes per-dataset fixed-width `.bin` shards
(`input_ids`, `prompt_mask`, `attention_mask`, `seq_lens`, `prompt_lens`) plus a
per-dataset `metadata.json` under `$PRETOKENIZED_ROOT/<dataset>/`.

### 2. Drop truncated rows and merge the manifest

Drop rows whose answer was truncated to fit `max_length` (a row is truncated iff
`seq_lens == max_length`):

```bash
python scripts/drop_truncated_rows.py --root "$PRETOKENIZED_ROOT"
```

Then rebuild the unified top-level `manifest.json` that the training dataloader
consumes (the parallel tokenize jobs race on this file, so it is rebuilt
deterministically here):

```bash
python scripts/merge_manifests.py --root "$PRETOKENIZED_ROOT"
```

Order matters: `drop_truncated_rows.py` rewrites each dataset's `.bin` shards and
per-dataset `metadata.json`, and `merge_manifests.py` must run **after** it so the
top-level `manifest.json` (which the dataloader reads) reflects the drops. (The
parallel tokenize jobs race on that top-level file, which is why it is rebuilt
here.)

After this, point training at `$PRETOKENIZED_ROOT` (the config reads it from
`PRETOKENIZED_ROOT`).

### Optional: corpus statistics

To inspect kept/dropped counts and token-length percentiles without writing
shards:

```bash
python -m flexmdm.data stats --config flexmdm/config/fsdp_train.yaml
```

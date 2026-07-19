<!--
This is the model card that ships INSIDE the released Hugging Face checkpoint
(as its README.md). scripts/prepare_release_checkpoint.py does not write it;
copy this file to the prepared checkpoint dir as README.md and fill in the
placeholders (repo id, GitHub URL, paper URL) before uploading.
-->
---
license: apache-2.0
base_model: Dream-org/Dream-Coder-v0-Base-7B
tags:
  - code
  - code-generation
  - diffusion
  - masked-diffusion
  - flexmdm
  - any-order
language:
  - en
---

# FlexMDM — Dream-Coder-7B

An **insertion + unmasking discrete-diffusion** model for Python code, produced by
fully fine-tuning [`Dream-org/Dream-Coder-v0-Base-7B`](https://huggingface.co/Dream-org/Dream-Coder-v0-Base-7B).
Unlike a fixed-length masked diffusion model, FlexMDM can **grow its sequence
during generation** (a learned insertion head) and unmask tokens in any order,
enabling genuinely any-order code generation.

- Paper: *From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models* —
  S. Kim\*, J. Kim\*, T. Lee\*, Y. Chen\*, Y. Du, S. Kakade, S. Chen (preprint, 2026). <!-- FILL IN: paper URL -->
- Code: [github.com/SeunggeunKimkr/genuine-any-order](https://github.com/SeunggeunKimkr/genuine-any-order), `FlexMDM/` subdirectory (training, data pipeline, inference, and evaluation).
- Checkpoint: `global_step_49500` (inference weights; optimizer/RNG state stripped).

## ⚠️ Loading (not a vanilla `AutoModel`)

This checkpoint hosts a Dream backbone plus FlexMDM-specific weights in
`flexmdm_extras.pt`. A bare `AutoModel.from_pretrained` returns only the backbone.
Load the full model with the FlexMDM package from the code repo:

```python
# 1) install the FlexMDM package:  pip install -e .  (in the repo's FlexMDM/ dir)
from huggingface_hub import snapshot_download
from flexmdm.utils import load_model_and_tokenizer

ckpt = snapshot_download("yuyuanchen0/flexmdm")
model, tokenizer = load_model_and_tokenizer(checkpoint_dir=ckpt, max_length=768)
```

`load_model_and_tokenizer` defaults to `attn_implementation="sdpa"` (works
everywhere). Pass `"flash_attention_2"` for speed (requires flash-attn), or
`"eager"` to bit-match the released evaluation traces.

For sampling (the ā = 2.9 inference-time schedule, temperature 0.1, 512 steps) use
`flexmdm.inference.flexmdm_generate`; see the code repo's `docs/REPRODUCE.md`.

## Model

- Base: `DreamModel` (7B; hidden 3584, 28 layers, GQA with 4 KV heads, vocab
  152064, diffusion mask id 151666), inherited unchanged.
- FlexMDM additions: a per-position **log-space insertion head**
  (`LayerNorm → Linear → GELU → Linear`, clamped to [-15, 15]) and **AdaLN time
  conditioning** on the insertion-progress coordinate.
- Schedules: **power** family, `α_t = 1−(1−t)^a` (insertion), `β_t = 1−(1−t)^(a·b)`
  (unmasking), with **a = b = 1.7**.
- Training: AdamW, LR 1e-5 (backbone) / 2e-5 (insertion head), global batch 576,
  max length 768, FSDP HYBRID_SHARD, 16× H100, ~3 days (checkpoint at step 49500).

## Training data

Fine-tuned on a five-source Hugging Face mixture (OpenCodeInstruct, opc-sft-stage2,
KodCode-V1-SFT-4o, and rStar-Coder seed/synthetic). **KodCode-V1-SFT-4o is
CC BY-NC 4.0 (non-commercial).** Full sources, filters, and licenses — and how to
reconstruct the tokenized set — are in the code repo's `docs/DATA.md`.

## Evaluation (pass@k; n = 16 samples/task)

| Benchmark | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|---|
| HumanEval  | 50.65 | 66.60 | 78.69 | 86.86 | 92.07 |
| HumanEval+ | 46.61 | 61.89 | 73.83 | 82.07 | 87.80 |
| MBPP       | 62.22 | 74.61 | 81.81 | 86.19 | 89.68 |
| MBPP+      | 52.86 | 64.38 | 72.04 | 77.00 | 80.16 |

These are the paper's Table 5 rows (extraction-robust any-of-4 grading, 30 s
test timeout). Generation is deterministic (content-addressed seeds), so the
sample set is bit-reproducible — see the code repo's
`evals/REPRODUCIBILITY.md` for the exact recipe.

FlexMDM also scores substantially higher than Dream-Coder on tree-based any-order
metrics (CBC/RUB/RUB+/OBW).

## License & intended use

Apache-2.0 (derived from Dream-Coder, Apache-2.0). Research artifact — **not a
deployment-ready system**; generated code may be incorrect or insecure, so sandbox
before executing. Note the non-commercial license on part of the training data
(above).

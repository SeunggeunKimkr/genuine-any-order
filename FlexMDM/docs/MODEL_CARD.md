# Model Card: FlexMDM (Dream-Coder-7B)

## Summary

FlexMDM is an **insertion + unmasking discrete-diffusion** model for Python code
generation, produced by fine-tuning **Dream-Coder-7B-Base**. It augments the
base model with a learned **insertion head** (so the sequence length can grow
during generation) and **AdaLN time conditioning**, and is trained with a
diffusion objective that combines token unmasking with token insertion. The
result supports any-order inference: tokens can be committed in any order and the
sequence length is decided as generation proceeds.

- **Distribution:** inference weights on the Hugging Face Hub (~16 GB; optimizer
  and RNG state stripped):
  [`yuyuanchen0/flexmdm`](https://huggingface.co/yuyuanchen0/flexmdm).
- **Intended use:** research on any-order / flexible-length code generation.
- **Not intended for:** production or deployment. Generated code may be incorrect
  or insecure; sandbox before executing.

## Base model

Fine-tuned from `Dream-org/Dream-Coder-v0-Base-7B` (Xie et al., 2025). The base
configuration (inherited unchanged by FlexMDM) is:

| Field | Value |
|---|---|
| Architecture | `DreamModel` |
| Hidden size | 3584 |
| Layers | 28 |
| Attention heads | 28 |
| KV heads (GQA) | 4 |
| Vocab size | 152064 |
| Diffusion mask token id | 151666 |
| Compute dtype | bfloat16 |

These values are from the released checkpoint's `config.json` (`architectures:
[DreamModel]`, `model_type: Dream`), inherited unchanged from
`Dream-org/Dream-Coder-v0-Base-7B`. The diffusion mask token id (151666) also
matches `tokenizer.mask_token_id`.

## FlexMDM additions

Both additions are implemented in `flexmdm/architecture.py`.

**Insertion head** — predicts a per-position log insertion rate:

```
LayerNorm -> Linear(hidden, hidden) -> GELU -> Linear(hidden, 1)
```

The scalar output is interpreted as the **log** of the per-position expected
insertion count; downstream code recovers the rate with `exp(.)`. The head output
is **clamped to [-15, 15]** for bf16 stability (`InsertionHead` in
`flexmdm/architecture.py`). (An earlier FlexMDM design predicted the nonnegative
rate directly via a Softplus; the released model predicts the log-rate, which the
paper reports is geometrically more natural and trains more stably.)

**AdaLN time conditioning** — the model is conditioned on the diffusion time
`t` (the raw time coordinate, in both training and inference — the released
checkpoint expects raw `t`, not a progress/alpha reparameterized coordinate):

- a sinusoidal embedding of the timestep `t`,
- an MLP: `Linear(hidden, 4*hidden) -> SiLU -> Linear(4*hidden, hidden)`,
- a **zero-initialized** AdaLN modulator on **each decoder layer's**
  `input_layernorm` and `post_attention_layernorm` (scale/shift applied as
  `(1 + scale) * norm(x) + shift`).

## Training objective

FlexMDM loss = **unmasking cross-entropy** (on masked answer tokens) +
**insertion Bregman loss** (on valid insertion gaps):

- The insertion Bregman loss uses the generating function `phi(x, y) = e^y - x*y`,
  where the head predicts `y` = log expected insertion count (verified in
  `flexmdm/trainer.py`: `phi(x, z) = exp(z) - x*z`, computed in float32).
- The two terms are summed to form the total loss.

## Schedules

FlexMDM uses the **power** schedule family (`flexmdm/schedules.py`):

- Insertion: `alpha_t = 1 - (1 - t)^a`
- Unmasking: `beta_t  = 1 - (1 - t)^(a*b)`

The released checkpoint uses **a = b = 1.7**, i.e. **insertion exponent 1.7** and
**unmasking exponent a*b = 2.89** (`flexmdm/config/fsdp_train.yaml`:
`insertion_schedule: power`, `unmasking_schedule: power`, `schedule_a: 1.7`,
`schedule_b: 1.7`).

## Training setup

Full fine-tune of all backbone + FlexMDM parameters.

| Setting | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01 |
| Peak LR | 1e-5 (backbone), 2e-5 (insertion head; `insertion_lr_multiplier=2.0`) |
| LR schedule | linear warmup (ratio 0.1) + cosine decay |
| Gradient clip | 1.0 |
| Global batch size | 576 |
| Max sequence length | 768 |
| Sequence formatting | prepend BOS; internal-pad prompt/answer separator |
| Sharding | FSDP HYBRID_SHARD (bf16 params, fp32 reduce) |
| Hardware | 16× H100 (80 GB), ~3 days (~1150 GPU-hours) |
| Steps | configured horizon 75000; released checkpoint = `global_step_49500` (checkpoints every 1500 steps) |

Data: a five-source Hugging Face mixture tokenized to length 768. See
[`DATA.md`](DATA.md) for the sources, filters, repeat factors, and licenses.

## Inference

The sampler (`flexmdm_generate` in `flexmdm/inference.py`; "Algorithm 1" in the
paper) alternates, per step:

- **confidence-based top-k unmasking**, and
- **Poisson insertion** (per-gap insertion counts sampled as
  `Poisson(rate * dt)`; `--insertion-count-sampler poisson`).

Published evaluation decoding: **512 steps**, max total sequence length
**768** (HumanEval) / **1100** (MBPP), **top-p 0.9**, **temperature 0.1**,
top-k confidence unmasking, eager attention.

**Temperature:** the released FlexMDM numbers use **temperature 0.1** on all
benchmarks (the paper's Appendix D.3 setting, motivated by the extra stochasticity
that Poisson insertion already introduces). These are the `MODEL=flexmdm`
defaults of `evals/humaneval_compare/run.sh` — no overrides needed. The
`MODEL=dreamcoder` **baseline** defaults instead follow Dream-Coder's own
convention (0.2 for HumanEval/HumanEval+, 0.1 for MBPP/MBPP+); the paper's DC
pass@k rows used temperature 1.0 with n=16 (see [`REPRODUCE.md`](REPRODUCE.md)).

### Inference-time schedule reparameterization (no retraining)

The model is **trained and evaluated in the raw-`t` convention**, but at inference
it is queried through a **more aggressive insertion schedule** without retraining.
For an inference insertion exponent `ā` and training exponent `a`, the model is
fed the training-time coordinate

```
tau(t) = 1 - (1 - t)^(ā / a)
```

The released evaluation uses **ā = 2.9** with `a = 1.7`, i.e.
`tau(t) = 1 - (1 - t)^(2.9/1.7)`. This is verified in `flexmdm/schedules.py`
(the reparameterization maps `model_t = train_alpha_inverse(inference_alpha(t))`,
which for power schedules reduces to the expression above). The corresponding
generation flags are given in [`REPRODUCE.md`](REPRODUCE.md).

## Evaluation

Benchmarks: HumanEval / HumanEval+ / MBPP / MBPP+ (pass@k), plus tree-based
any-order metrics (CBC, RUB, RUB+, OBW). Headline numbers and how to reproduce
them are in [`REPRODUCE.md`](REPRODUCE.md).

## Limitations and risks

- Research artifact only; **not** hardened for deployment.
- Generated code may be **incorrect or insecure** — always sandbox before running.
- Trained on Python SFT corpora; behavior outside that distribution is untested.
- Uses training data that includes a **non-commercial** dataset
  (KodCode-V1-SFT-4o, CC BY-NC 4.0); see [`DATA.md`](DATA.md) and [`../NOTICE`](../NOTICE).

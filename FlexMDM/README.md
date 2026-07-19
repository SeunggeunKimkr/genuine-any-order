# FlexMDM: Any-Order Code Generation from Dream-Coder

Public code release for the **FlexMDM** (Flexible-length Masked Diffusion Model)
code-generation model from the paper:

> **From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models.**
> Seunggeun Kim\*, Jaeyeon Kim\*, Taekyun Lee\*, Yuyuan Chen\*, Yilun Du,
> Sham Kakade, Sitan Chen (\*co-first authors).
> Preprint, 2026.
> <!-- FILL IN: arXiv URL. -->

```bibtex
@article{kim2026interface,
  title   = {From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models},
  author  = {Kim, Seunggeun and Kim, Jaeyeon and Lee, Taekyun and Chen, Yuyuan
             and Du, Yilun and Kakade, Sham and Chen, Sitan},
  journal = {arXiv preprint},
  year    = {2026}
}
```

FlexMDM fine-tunes **Dream-Coder-7B-Base** into an **insertion + unmasking**
discrete-diffusion model for Python code. Unlike a fixed-length masked diffusion
model, FlexMDM can grow its sequence during generation (via a learned insertion
head) and unmask tokens in any order, enabling genuinely any-order inference.

## Scope of this repository

This repository is **only** the FlexMDM (code-generation) portion of the paper.
The paper's **LatentMDM** (§4.2) and its **§3 similarity analysis** are separate
work and are **not** included here.

Built on:

- **FlexMDM** — Kim et al., 2025a, *Any-order flexible length masked diffusion*,
  [arXiv:2509.01025](https://arxiv.org/abs/2509.01025).
- **Dream-Coder 7B** — Xie et al., 2025,
  [arXiv:2509.01142](https://arxiv.org/abs/2509.01142).

## Repository layout

```
flexmdm/            Model + training code
  architecture.py     Dream-Coder backbone + FlexMDM insertion head & AdaLN time conditioning
  trainer.py          FlexMDM loss (unmasking cross-entropy + insertion Bregman loss)
  schedules.py        Insertion/unmasking schedules (linear, quadratic, power, log_linear, logit_power)
  inference.py        FlexMDM sampler (flexmdm_generate) + inference-time schedule reparameterization
  data.py             Dataset filtering + tokenization pipeline (`python -m flexmdm.data`)
  train_fsdp.py       FSDP HYBRID_SHARD training entrypoint (Hydra config)
  config/             fsdp_train.yaml (the released training recipe)
evals/              Evaluation pipeline
  humaneval_compare/  Generation + pass@k on HumanEval/HumanEval+/MBPP/MBPP+
  tree_analysis/      Any-order metrics (CBC / RUB / RUB+ / OBW)
  run_full_eval.sh    End-to-end driver: generate -> pass@k -> any-order
  REPRODUCIBILITY.md  Bit-exact reproduction of the released pass@k tables
scripts/            Data + training launchers (pretokenize, merge, drop-truncated, train_16gpu)
docs/               MODEL_CARD.md, DATA.md, REPRODUCE.md, CLUSTER.md, hf_checkpoint_README.md
paths.env.template  Copy to paths.env and edit paths/accounts for your machine
LICENSE             Apache-2.0
NOTICE              Attribution for Dream-Coder and the training datasets
```

## Released checkpoint

- **Checkpoint:** `global_step_49500`.
- Published on the Hugging Face Hub as **inference weights** (~16 GB; optimizer
  and RNG state stripped):
  [`yuyuanchen0/flexmdm`](https://huggingface.co/yuyuanchen0/flexmdm).

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full model description and
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for evaluation numbers and how to
reproduce them.

## Quickstart

This directory is the self-contained FlexMDM codebase — run every command from
it, and read "repository root" in the docs as this `FlexMDM/` directory.

1. Set up the environment and a `paths.env` — see [`docs/CLUSTER.md`](docs/CLUSTER.md).
2. Rebuild the tokenized training data — see [`docs/DATA.md`](docs/DATA.md).
3. Train (`scripts/train_16gpu.sh`) or download the released checkpoint.
4. Generate and score — see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Training-data note (please read)

The released checkpoint was trained on a five-source mixture that
**includes rStar-Coder and KodCode-V1-SFT-4o** — the latter licensed
**CC BY-NC 4.0 (non-commercial)**. The exact sources, filters, repeat factors,
and licenses are in [`docs/DATA.md`](docs/DATA.md).

## License

This repository's own code is released under the **Apache-2.0** license
(see [`LICENSE`](LICENSE)). It derives from Dream-Coder (Apache-2.0) and uses
several third-party datasets under their own licenses — **including one
non-commercial dataset (KodCode-V1-SFT-4o, CC BY-NC 4.0)**. Read
[`NOTICE`](NOTICE) and [`docs/DATA.md`](docs/DATA.md) before use, and accept each
data source's terms.

## Safety

FlexMDM is a **research artifact, not a deployment-ready system**. Generated code
may be incorrect or insecure — always run it in a sandbox before executing.

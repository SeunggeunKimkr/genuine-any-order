<div align="center">

# From Interface to Inference

### Eliciting Any-Order Inference from Any-Order Models

<br>

[Seunggeun Kim](https://seunggeunkimkr.github.io)<sup>1,\*</sup> &nbsp;&middot;&nbsp;
[Jaeyeon Kim](https://jaeyeonkim01.github.io)<sup>2,\*</sup> &nbsp;&middot;&nbsp;
[Taekyun Lee](https://taekyunl.github.io)<sup>1,\*</sup> &nbsp;&middot;&nbsp;
[Yuyuan Chen](https://yuyuanchen0.github.io)<sup>2,\*</sup>

Yilun Du<sup>2</sup> &nbsp;&middot;&nbsp;
Sham Kakade<sup>2</sup> &nbsp;&middot;&nbsp;
Sitan Chen<sup>2</sup>

<sup>1</sup> The University of Texas at Austin  <sup>2</sup> Harvard University  <sup>\*</sup> Co-first authors

<br>

</div>

> MDMs promise any-order generation but often collapse to left-to-right decoding. We introduce **LatentMDM** and **FlexMDM** that enable genuinely any-order inference.

## Repository structure

| Directory | Contents |
|---|---|
| [`FlexMDM/`](FlexMDM/) | FlexMDM fine-tuned from Dream-Coder-7B-Base for any-order Python code generation: training, data pipeline, inference, and the full evaluation suite (pass@k + any-order metrics). Self-contained, with its own docs and environment. |
| [`LatentMDM/`](LatentMDM/) | LatentMDM (LP-MDM) training, sampling, and evaluation on TinyGSM. |

**Released checkpoint:** the FlexMDM model is on the Hugging Face Hub at
[`yuyuanchen0/flexmdm`](https://huggingface.co/yuyuanchen0/flexmdm)
(inference weights; see [`FlexMDM/README.md`](FlexMDM/README.md) for loading
and for reproducing the paper's tables).

`FlexMDM/` is licensed under Apache-2.0 (see [`FlexMDM/LICENSE`](FlexMDM/LICENSE)
and [`FlexMDM/NOTICE`](FlexMDM/NOTICE)).

## Acknowledgments

This codebase builds on [PUMA](https://github.com/JaeyeonKim01/PUMA) and [FlexMDM](https://github.com/brianlck/FlexMDM).

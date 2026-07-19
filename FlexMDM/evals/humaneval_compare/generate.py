"""Unified driver that generates code completions + trajectories.

Supports both HumanEval and MBPP families:
  --dataset humaneval | humaneval_plus | mbpp | mbpp_plus

The HE / HE+ pair share the same generation prompts (and thus the same
traces); the +-variant differs only in the test code used at pass@k time.
Same for MBPP / MBPP+. So this script generates under the underlying
"generation dataset" (humaneval or mbpp), regardless of which variant the
user asked for.

Usage (FlexMDM; --algs defaults to top_k, the published config — see
evals/REPRODUCIBILITY.md for the full bit-exact command):
    python -m evals.humaneval_compare.generate \\
        --model flexmdm --checkpoint-dir <path> \\
        --dataset humaneval --output-root <dir> \\
        --n-samples 16 --steps 512 \\
        --temperature 0.1 --max-length 768   # mbpp: --max-length 1100

Usage (Dream-Coder base; --algs defaults to entropy):
    python -m evals.humaneval_compare.generate \\
        --model dreamcoder \\
        --dreamcoder-model Dream-org/Dream-Coder-v0-Base-7B \\
        --dataset humaneval --output-root <dir> \\
        --n-samples 16 --steps 512

Per-(task, alg) work unit produces ``n-samples`` samples in batched calls,
saving one .pt record per (task, alg, sample) plus an HTML viewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Optional

import torch

from flexmdm.inference import build_meaningful_token_mask, flexmdm_generate
from flexmdm.utils import load_model_and_tokenizer as load_flexmdm_model
from transformers import AutoModel, AutoTokenizer

from .common import (
    ALL_DATASETS,
    GEN_HUMANEVAL,
    Task,
    dreamcoder_prompt_style_for,
    encode_dreamcoder_prompt,
    encode_flexmdm_prompt,
    gen_dataset_for,
    html_path,
    load_saved_tasks,
    load_tasks,
    record_path,
    save_tasks,
    shard,
    stack_attention,
    stack_trajectory,
    subset_tasks,
    tasks_path,
)
from .render_html import render_record_to_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["flexmdm", "dreamcoder"])
    p.add_argument("--dataset", required=True, choices=list(ALL_DATASETS),
                   help="CLI dataset; HE/HE+ generate identical traces, as do "
                        "MBPP/MBPP+. The +-variants only differ at pass@k.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Required when --model=flexmdm.")
    p.add_argument("--dreamcoder-model",
                   default="Dream-org/Dream-Coder-v0-Base-7B",
                   help="HF repo id or local path for Dream-Coder base.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tasks-file", default=None,
                   help="Optional tasks.json from a prior run. If absent, "
                        "loaded fresh from HF and written to "
                        "<output_root>/<gen_dataset>/tasks.json.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, run only the first N tasks (sanity check).")
    p.add_argument("--stratified", action="store_true",
                   help="With --limit N, pick N stratified tasks across the "
                        "full set instead of the first N.")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Max samples per GPU call. Total n_samples is processed "
                        "in ceil(n_samples / batch_size) chunks.")
    p.add_argument("--algs", nargs="+", default=None,
                   help="Decoding/confidence algs. Default: top_k for "
                        "--model flexmdm, entropy for --model dreamcoder "
                        "(the published configurations).")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-length", type=int, default=768,
                   help="Total sequence length cap (FlexMDM). For Dream-Coder, "
                        "max_new_tokens = max_length - prompt_len, unless "
                        "--max-new-tokens is set. Published runs: 768 for "
                        "HumanEval, 1100 for MBPP.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Dream-Coder only. Fixes the unmasked region size; "
                        "total seq len becomes prompt_len + max_new_tokens.")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default=None,
                   help="Attention backend. Default for --model flexmdm is "
                        "'eager' (the bit-exact published setting; see "
                        "evals/REPRODUCIBILITY.md); sdpa/flash_attention_2 "
                        "are faster but not bit-reproducible. Dream-Coder "
                        "uses the model's own default unless set.")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--insertion-schedule", default="linear")
    p.add_argument("--unmasking-schedule", default="quadratic")
    p.add_argument("--insertion-exponent", type=float, default=None,
                   help="FlexMDM only. Required when --insertion-schedule=power. "
                        "For the (a,b) parametrization a = insertion_exponent.")
    p.add_argument("--unmasking-exponent", type=float, default=None,
                   help="FlexMDM only. Required when --unmasking-schedule=power. "
                        "For the (a,b) parametrization unmasking_exponent = a*b.")
    p.add_argument("--insertion-params", type=str, default=None,
                   help="FlexMDM only. JSON dict of schedule params for non-power "
                        "schedules. Examples: log_linear -> "
                        "'{\"lam\": 1.0, \"c\": 50.0}'; logit_power -> "
                        "'{\"a\": 0.5, \"b\": 2.5}'.")
    p.add_argument("--unmasking-params", type=str, default=None,
                   help="FlexMDM only. JSON dict of schedule params for non-power "
                        "schedules (see --insertion-params).")
    p.add_argument("--train-insertion-schedule", default=None,
                   help="FlexMDM only. Schedule used at training; defaults to "
                        "--insertion-schedule. Used by the schedule "
                        "reparameterization so the model is queried at training-time "
                        "t-coordinates even when inference schedule differs.")
    p.add_argument("--train-insertion-exponent", type=float, default=None,
                   help="FlexMDM only. Insertion exponent used at training; "
                        "only matters when it differs from --insertion-exponent. "
                        "Triggers schedule reparameterization.")
    p.add_argument("--train-insertion-params", type=str, default=None,
                   help="FlexMDM only. JSON dict of training-side insertion "
                        "schedule params (only matters when train schedule differs "
                        "from inference schedule).")
    p.add_argument("--insertion-count-sampler", default="poisson",
                   choices=["poisson", "floor_bernoulli"],
                   help="FlexMDM only: how integer insertion counts are drawn "
                        "from the per-gap rate (poisson is default; "
                        "floor_bernoulli is a deterministic alternative).")
    p.add_argument("--use-midpoint-hazard-rate", action="store_true", default=False,
                   help="FlexMDM only. Evaluate the insertion and unmasking "
                        "hazard rates at t_k + dt/2 (midpoint quadrature) "
                        "instead of the default left endpoint t_k. Does not "
                        "affect model_t (the model is still queried at t_k).")
    p.add_argument("--force-unmask-long-segments", action="store_true",
                   default=False)
    p.add_argument("--force-unmask-min-segment-len", type=int, default=9)
    p.add_argument("--force-unmask-top-k-per-position", type=int, default=10)
    p.add_argument("--meaningful-replace", action="store_true", default=False,
                   help="FlexMDM only. Replace the lowest-prob ordinary unmask "
                        "with a meaningful-token candidate from a long mask "
                        "block when its summed window mass exceeds the lowest "
                        "ordinary commit's prob.")
    p.add_argument("--meaningful-replace-min-segment-len", type=int, default=20,
                   help="Minimum mask-run length L for the meaningful-replace "
                        "trick to fire on that run.")
    p.add_argument("--meaningful-replace-window-len", type=int, default=5,
                   help="Sliding-window length over the non-boundary region "
                        "for the meaningful-replace trick.")
    p.add_argument("--meaningful-replace-min-step", type=int, default=100,
                   help="The meaningful-replace trick only fires from this "
                        "step index onward (default 100).")
    p.add_argument("--meaningful-exclude-strings", type=str, default="",
                   help="Comma-separated list of token strings (post-strip) "
                        "to also force-not-meaningful, on top of the default "
                        "whitespace+punct filter. Useful for blocking stub "
                        "builtins like 'print' from being committed by the "
                        "meaningful-replace trick.")
    p.add_argument("--tokenizer-source", default=None,
                   help="Override tokenizer for HTML rendering. Defaults to "
                        "the model's own tokenizer source.")
    args = p.parse_args()
    if args.model == "flexmdm" and not args.checkpoint_dir:
        p.error("--checkpoint-dir is required when --model=flexmdm")
    return args


def _parse_params(s):
    if s is None:
        return None
    return json.loads(s)


def stable_seed(task_id: str, model: str, alg: str) -> int:
    h = hashlib.sha256(f"{task_id}|{model}|{alg}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name!r}")
    return dtype


def ensure_tasks(args: argparse.Namespace, gen_dataset: str) -> list[Task]:
    """Load (or compute) the task list, and persist it under the output root."""
    tasks_json = args.tasks_file or tasks_path(args.output_root, gen_dataset)
    if os.path.isfile(tasks_json):
        tasks = load_saved_tasks(tasks_json)
        print(f"[tasks] loaded {len(tasks)} from {tasks_json}", flush=True)
    else:
        tasks = load_tasks(args.dataset)
        save_tasks(tasks_json, tasks)
        print(f"[tasks] saved {len(tasks)} {gen_dataset} tasks "
              f"to {tasks_json}", flush=True)
    if args.limit is not None:
        tasks = subset_tasks(tasks, limit=args.limit, stratified=args.stratified)
        print(f"[tasks] limited to {len(tasks)} "
              f"({'stratified' if args.stratified else 'first'})", flush=True)
    return tasks


def chunk_already_done(
    *, args: argparse.Namespace, task: Task, alg: str, gen_dataset: str,
    model: str, start: int, end: int,
) -> bool:
    """True iff every sample in [start, end) already has a record + html.

    Used to extend or resume an existing run in place: chunks whose
    samples were already generated are skipped, and because the RNG is seeded
    per chunk as ``base_seed + start``, freshly generated chunks draw from
    seeds the original run never used.
    """
    for k in range(start, end):
        rp = record_path(args.output_root, gen_dataset, task.task_id,
                         model, alg, k)
        hp = html_path(args.output_root, gen_dataset, task.task_id,
                       model, alg, k)
        if not (os.path.isfile(rp) and os.path.isfile(hp)):
            return False
    return True


def run_flexmdm_unit(
    *,
    task: Task,
    alg: str,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tokenizer_source: str,
    gen_dataset: str,
    meaningful_token_mask: Optional[torch.Tensor] = None,
) -> None:
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    input_ids_1, attention_mask_1, prompt_mask_1, boundary = encode_flexmdm_prompt(
        tokenizer,
        task.prompt,
        max_length=args.max_length,
        pad_id=pad_id,
    )

    meta_base = {
        "task_id": task.task_id,
        "task_index": task.index,
        "entry_point": task.entry_point,
        "dataset": gen_dataset,
        "model": "flexmdm",
        "alg": alg,
        "steps": int(args.steps),
        "temperature": float(args.temperature),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "insertion_schedule": args.insertion_schedule,
        "unmasking_schedule": args.unmasking_schedule,
        "insertion_exponent": args.insertion_exponent,
        "unmasking_exponent": args.unmasking_exponent,
        "insertion_params": args.insertion_params,
        "unmasking_params": args.unmasking_params,
        "train_insertion_schedule": args.train_insertion_schedule,
        "train_insertion_exponent": args.train_insertion_exponent,
        "train_insertion_params": args.train_insertion_params,
        "insertion_count_sampler": args.insertion_count_sampler,
        "use_midpoint_hazard_rate": bool(args.use_midpoint_hazard_rate),
        "force_unmask_long_segments": bool(args.force_unmask_long_segments),
        "force_unmask_min_segment_len": int(args.force_unmask_min_segment_len),
        "force_unmask_top_k_per_position": int(args.force_unmask_top_k_per_position),
        "meaningful_replace": bool(args.meaningful_replace),
        "meaningful_replace_min_segment_len": int(args.meaningful_replace_min_segment_len),
        "meaningful_replace_window_len": int(args.meaningful_replace_window_len),
        "meaningful_replace_min_step": int(args.meaningful_replace_min_step),
        "meaningful_exclude_strings": args.meaningful_exclude_strings,
        "checkpoint_dir": args.checkpoint_dir,
        "tokenizer_source": tokenizer_source,
    }

    base_seed = stable_seed(task.task_id, "flexmdm", alg)
    for start in range(0, args.n_samples, args.batch_size):
        end = min(start + args.batch_size, args.n_samples)
        chunk_n = end - start
        if args.skip_existing and chunk_already_done(
            args=args, task=task, alg=alg, gen_dataset=gen_dataset,
            model="flexmdm", start=start, end=end,
        ):
            print(
                f"[skip] {gen_dataset}/{task.task_id} alg={alg} "
                f"samples[{start}:{end}] already exist",
                flush=True,
            )
            continue
        input_ids = input_ids_1.repeat(chunk_n, 1).to(device)
        attention_mask = attention_mask_1.repeat(chunk_n, 1).to(device)
        prompt_mask = prompt_mask_1.repeat(chunk_n, 1).to(device)

        torch.manual_seed(base_seed + start)
        t0 = time.time()
        out = flexmdm_generate(
            model,
            steps=args.steps,
            input_ids=input_ids,
            mask_id=mask_id,
            pad_id=pad_id,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            temperature=args.temperature,
            confidence_method=alg,
            trace=True,
            insertion_schedule=args.insertion_schedule,
            unmasking_schedule=args.unmasking_schedule,
            insertion_exponent=args.insertion_exponent,
            unmasking_exponent=args.unmasking_exponent,
            insertion_params=_parse_params(args.insertion_params),
            unmasking_params=_parse_params(args.unmasking_params),
            train_insertion_schedule=args.train_insertion_schedule,
            train_insertion_exponent=args.train_insertion_exponent,
            train_insertion_params=_parse_params(args.train_insertion_params),
            insertion_count_sampler=args.insertion_count_sampler,
            force_unmask_long_segments=args.force_unmask_long_segments,
            force_unmask_min_segment_len=args.force_unmask_min_segment_len,
            force_unmask_top_k_per_position=args.force_unmask_top_k_per_position,
            meaningful_replace=args.meaningful_replace,
            meaningful_replace_min_segment_len=args.meaningful_replace_min_segment_len,
            meaningful_replace_window_len=args.meaningful_replace_window_len,
            meaningful_replace_min_step=args.meaningful_replace_min_step,
            meaningful_token_mask=meaningful_token_mask,
            use_midpoint_hazard_rate=args.use_midpoint_hazard_rate,
        )
        elapsed = time.time() - t0
        print(
            f"[flex] {gen_dataset}/{task.task_id} alg={alg} "
            f"samples[{start}:{end}] steps={args.steps} took {elapsed:.1f}s",
            flush=True,
        )

        sequences = stack_trajectory(out["history"])
        attn_masks = stack_attention(out["attention_mask_history"])
        insertion_masks_chunk = stack_attention(out["insertion_history"])

        for i in range(chunk_n):
            k = start + i
            rec_path = record_path(
                args.output_root, gen_dataset, task.task_id, "flexmdm", alg, k
            )
            _save_with_insertion_masks(
                rec_path,
                prompt=task.prompt,
                prompt_len=boundary,
                sequences=sequences[:, i, :].contiguous(),
                attention_masks=attn_masks[:, i, :].contiguous(),
                insertion_masks=insertion_masks_chunk[:, i, :].contiguous(),
                mask_id=mask_id,
                pad_id=pad_id,
                meta={**meta_base, "sample_k": int(k),
                      "insertion_masks_stored": True},
            )
            out_html = html_path(
                args.output_root, gen_dataset, task.task_id, "flexmdm", alg, k
            )
            render_record_to_html(rec_path, out_html, tokenizer=tokenizer)
        del out, sequences, attn_masks, insertion_masks_chunk
        torch.cuda.empty_cache()


def run_dreamcoder_unit(
    *,
    task: Task,
    alg: str,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    tokenizer_source: str,
    gen_dataset: str,
) -> None:
    prompt_style = dreamcoder_prompt_style_for(gen_dataset)
    input_ids_1, attention_mask_1, prompt_len = encode_dreamcoder_prompt(
        tokenizer, task.prompt, add_bos=True, prompt_style=prompt_style
    )
    if args.max_new_tokens is not None:
        max_new_tokens = int(args.max_new_tokens)
    else:
        max_new_tokens = args.max_length - prompt_len
        if max_new_tokens <= 0:
            raise ValueError(
                f"max_length {args.max_length} <= prompt_len {prompt_len} for "
                f"task {task.task_id}; increase --max-length or set "
                f"--max-new-tokens."
            )
    mask_id = int(tokenizer.mask_token_id)
    pad_id = int(
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    total_len = prompt_len + max_new_tokens
    init_full = torch.full((1, total_len), mask_id, dtype=torch.long)
    init_full[0, :prompt_len] = input_ids_1[0]

    meta_base = {
        "task_id": task.task_id,
        "task_index": task.index,
        "entry_point": task.entry_point,
        "dataset": gen_dataset,
        "model": "dreamcoder",
        "alg": alg,
        "steps": int(args.steps),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(max_new_tokens),
        "max_length": int(args.max_length),
        "batch_size": int(args.batch_size),
        "dreamcoder_model": args.dreamcoder_model,
        "dreamcoder_prompt_style": prompt_style,
        "tokenizer_source": tokenizer_source,
    }

    base_seed = stable_seed(task.task_id, "dreamcoder", alg)
    for start in range(0, args.n_samples, args.batch_size):
        end = min(start + args.batch_size, args.n_samples)
        chunk_n = end - start
        input_ids = input_ids_1.repeat(chunk_n, 1).to(device)
        attention_mask = attention_mask_1.repeat(chunk_n, 1).to(device)

        torch.manual_seed(base_seed + start)
        t0 = time.time()
        gen = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=None,
            alg=alg,
            alg_temp=0.0,
        )
        elapsed = time.time() - t0
        print(
            f"[dream] {gen_dataset}/{task.task_id} alg={alg} "
            f"samples[{start}:{end}] steps={args.steps} took {elapsed:.1f}s",
            flush=True,
        )

        init_replicated = init_full.repeat(chunk_n, 1).to(device)
        history: list[torch.Tensor] = [init_replicated] + list(gen.history)
        attn_full = torch.ones(
            (chunk_n, total_len), dtype=torch.bool, device=device
        )
        attention_masks_history = [attn_full for _ in history]
        insertion_masks_history = [
            torch.zeros((chunk_n, total_len), dtype=torch.bool, device=device)
            for _ in history
        ]

        sequences = stack_trajectory(history)
        attn_masks = stack_attention(attention_masks_history)
        insertion_masks_chunk = stack_attention(insertion_masks_history)

        for i in range(chunk_n):
            k = start + i
            rec_path = record_path(
                args.output_root, gen_dataset, task.task_id, "dreamcoder",
                alg, k,
            )
            _save_with_insertion_masks(
                rec_path,
                prompt=task.prompt,
                prompt_len=prompt_len,
                sequences=sequences[:, i, :].contiguous(),
                attention_masks=attn_masks[:, i, :].contiguous(),
                insertion_masks=insertion_masks_chunk[:, i, :].contiguous(),
                mask_id=mask_id,
                pad_id=pad_id,
                meta={**meta_base, "sample_k": int(k),
                      "insertion_masks_stored": True},
            )
            out_html = html_path(
                args.output_root, gen_dataset, task.task_id, "dreamcoder",
                alg, k,
            )
            render_record_to_html(rec_path, out_html, tokenizer=tokenizer)
        del gen, sequences, attn_masks, insertion_masks_chunk
        torch.cuda.empty_cache()


def _save_with_insertion_masks(
    path: str,
    *,
    prompt: str,
    prompt_len: int,
    sequences: torch.Tensor,
    attention_masks: torch.Tensor,
    insertion_masks: torch.Tensor,
    mask_id: int,
    pad_id: int,
    meta: dict,
) -> None:
    """Save a record including the insertion_masks tensor."""
    from .common import save_record
    save_record(
        path,
        prompt=prompt,
        prompt_len=prompt_len,
        sequences=sequences,
        attention_masks=attention_masks,
        mask_id=mask_id,
        pad_id=pad_id,
        meta=meta,
    )
    existing = torch.load(path, map_location="cpu", weights_only=False)
    existing["insertion_masks"] = insertion_masks.contiguous()
    torch.save(existing, path)


def unit_already_done(
    *, args: argparse.Namespace, task: Task, alg: str, gen_dataset: str,
) -> bool:
    for k in range(args.n_samples):
        rp = record_path(args.output_root, gen_dataset, task.task_id,
                         args.model, alg, k)
        hp = html_path(args.output_root, gen_dataset, task.task_id,
                       args.model, alg, k)
        if not (os.path.isfile(rp) and os.path.isfile(hp)):
            return False
    return True


def main() -> None:
    args = parse_args()
    if args.algs is None:
        # Published configurations: FlexMDM decodes with top_k confidence,
        # the Dream-Coder baseline with negative entropy.
        args.algs = ["top_k"] if args.model == "flexmdm" else ["entropy"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda", 0)

    gen_dataset = gen_dataset_for(args.dataset)
    os.makedirs(args.output_root, exist_ok=True)
    tasks = ensure_tasks(args, gen_dataset)

    units: list[tuple[Task, str]] = [
        (task, alg) for task in tasks for alg in args.algs
    ]
    my_units = shard(units, args.shard_index, args.num_shards)
    print(
        f"[shard] {args.shard_index}/{args.num_shards}: "
        f"{len(my_units)} units out of {len(units)} total "
        f"({gen_dataset}, model={args.model})",
        flush=True,
    )

    if args.model == "flexmdm":
        print(f"[load] FlexMDM from {args.checkpoint_dir}", flush=True)
        model, tokenizer = load_flexmdm_model(
            checkpoint_dir=args.checkpoint_dir,
            max_length=args.max_length,
            torch_dtype_name=args.torch_dtype,
            attn_implementation=args.attn_implementation or "eager",
            trust_remote_code=True,
        )
        model.eval().to(device)
        tokenizer_source = args.tokenizer_source or args.checkpoint_dir
        meaningful_token_mask: Optional[torch.Tensor] = None
        if args.meaningful_replace and args.model == "flexmdm":
            # The mask must match the model's logits last-dim, which can be
            # padded beyond len(tokenizer) (e.g. Qwen/Dream pads to multiples
            # of 64 for tensor cores). Probe the wrapped backbone's config.
            vocab_size = 0
            for obj in (model, getattr(model, "backbone", None)):
                if obj is None:
                    continue
                cfg = getattr(obj, "config", None)
                if cfg is not None:
                    vs = int(getattr(cfg, "vocab_size", 0) or 0)
                    if vs > 0:
                        vocab_size = vs
                        break
            if vocab_size == 0:
                vocab_size = len(tokenizer)
            print(
                f"[mr] building meaningful-token mask over vocab_size={vocab_size} "
                f"(tokenizer len={len(tokenizer)})",
                flush=True,
            )
            extra_strings = [
                s.strip() for s in (args.meaningful_exclude_strings or "").split(",")
                if s.strip()
            ]
            meaningful_token_mask = build_meaningful_token_mask(
                tokenizer, vocab_size=vocab_size,
                extra_excluded_strings=extra_strings or None,
                device=device,
            )
            print(
                f"[mr] meaningful_token_mask: "
                f"{int(meaningful_token_mask.sum().item())}/"
                f"{vocab_size} marked meaningful "
                f"(extra_excluded_strings={extra_strings})",
                flush=True,
            )
    else:
        print(f"[load] Dream-Coder from {args.dreamcoder_model}", flush=True)
        dtype = resolve_dtype(args.torch_dtype)
        kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        model = AutoModel.from_pretrained(
            args.dreamcoder_model, **kwargs
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.dreamcoder_model, trust_remote_code=True
        )
        tokenizer_source = args.tokenizer_source or args.dreamcoder_model
        meaningful_token_mask = None

    start_time = time.time()
    for i, (task, alg) in enumerate(my_units):
        if args.skip_existing and unit_already_done(
            args=args, task=task, alg=alg, gen_dataset=gen_dataset
        ):
            print(
                f"[skip] ({i+1}/{len(my_units)}) {task.task_id} {alg}: "
                f"all samples present",
                flush=True,
            )
            continue
        print(
            f"[run ] ({i+1}/{len(my_units)}) {args.model} {gen_dataset}/"
            f"{task.task_id} {alg} n={args.n_samples}",
            flush=True,
        )
        if args.model == "flexmdm":
            run_flexmdm_unit(
                task=task, alg=alg, args=args,
                model=model, tokenizer=tokenizer,
                device=device, tokenizer_source=tokenizer_source,
                gen_dataset=gen_dataset,
                meaningful_token_mask=meaningful_token_mask,
            )
        else:
            run_dreamcoder_unit(
                task=task, alg=alg, args=args,
                model=model, tokenizer=tokenizer,
                device=device, tokenizer_source=tokenizer_source,
                gen_dataset=gen_dataset,
            )
    print(
        f"[done] shard {args.shard_index}/{args.num_shards}: "
        f"{len(my_units)} units in {time.time() - start_time:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)

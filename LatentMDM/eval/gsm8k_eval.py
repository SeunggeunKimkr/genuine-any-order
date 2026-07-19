import torch
import os
import io
import contextlib
import signal
import numpy as np
import math
import time
import torch.distributed as dist
import re
import json
import warnings
from transformers import AutoTokenizer
from omegaconf import ListConfig
from sampling import mdm_sampling, mdm_sampling_block, arm_sampling, lpmdm_sampling
from tqdm import tqdm
from datasets import load_dataset


# -----------------------------
# Tokenizer cache
# -----------------------------
def get_tokenizer(tokenizer_name: str | None = None):
    return AutoTokenizer.from_pretrained(tokenizer_name or TOKENIZER_NAME, use_fast=True)

def get_sep_ids(tokenizer_name: str | None = None):
    tok = get_tokenizer(tokenizer_name)
    return tok(SEP, add_special_tokens=False).input_ids

TOKENIZER_NAME = "Qwen/Qwen2-0.5B"
MAX_LEN = 512
SEP = "\n"
MASK_ID = 151644

# SEP__ID: 198
# PAD__ID: 151643
# EOS__ID: 151643
# model_vocab_size: 151645

def _cfg_value(container, key):
    if container is None:
        return None
    if hasattr(container, "get"):
        return container.get(key, None)
    return getattr(container, key, None)

def _tokenizer_name_from_cfg(cfg):
    for section_name in ("validation", "data"):
        section = getattr(cfg, section_name, None)
        for key in ("tokenizer_name", "tokenizer"):
            value = _cfg_value(section, key)
            if value not in (None, "", "none", "None"):
                return str(value)

    data_cfg = getattr(cfg, "data", None)
    data_dir = _cfg_value(data_cfg, "data_dir")
    if data_dir not in (None, "", "none", "None"):
        meta_path = os.path.join(str(data_dir), "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            tokenizer_name = meta.get("tokenizer")
            if tokenizer_name not in (None, "", "none", "None"):
                return str(tokenizer_name)

    return TOKENIZER_NAME

def _gsm8k_cache_path(tokenizer_name: str, mask_id: int):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", tokenizer_name).strip("_")
    if not slug:
        slug = "tokenizer"
    return os.path.join("data", "gsm8k_test", f"test_mdm_{slug}_mask{int(mask_id)}.json")

# -----------------------------
# GSM8K answer parsing
# -----------------------------

def _is_sequence_config(value):
    return isinstance(value, (list, tuple, ListConfig))

def _as_list(value):
    if _is_sequence_config(value):
        return list(value)
    return [value]

def _validation_limit(cfg):
    limit = cfg.validation.get("limit", None)
    if isinstance(limit, str) and limit.lower() in ("", "none", "null"):
        return None
    if limit is None:
        return None
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"validation.limit must be positive or null, got {limit}.")
    return limit

def _ids_to_list(ids):
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().view(-1).tolist()]
    if isinstance(ids, np.ndarray):
        return [int(x) for x in ids.reshape(-1).tolist()]
    return [int(x) for x in ids]

def _tokenizer_pad_id(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    return int(pad_id)

def _decode_token_ids(tokenizer, token_ids):
    if len(token_ids) == 0:
        return ""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return tokenizer.batch_decode([token_ids], skip_special_tokens=True)[0]

def _decode_prompt(tokenizer, input_ids, mask_id: int):
    token_ids = _ids_to_list(input_ids)
    try:
        first_mask = token_ids.index(int(mask_id))
        token_ids = token_ids[:first_mask]
    except ValueError:
        pass
    return _decode_token_ids(tokenizer, token_ids)

def _json_score(score):
    score = float(score)
    if math.isfinite(score):
        return score
    return "-inf"

def _decoding_order_from_trace(track_steps, sample_idx: int, max_seg_num: int):
    order = []
    seen = set()
    for step in track_steps:
        chosen_slot = torch.as_tensor(step["chosen_slot"])
        chosen = int(chosen_slot[sample_idx].item())
        if 0 <= chosen < max_seg_num and chosen not in seen:
            order.append(chosen + 1)
            seen.add(chosen)

    for slot_idx in range(max_seg_num):
        if slot_idx not in seen:
            order.append(slot_idx + 1)
    return order

def _lpmdm_trace_record(
    *,
    index: int,
    input_ids,
    answer: str,
    prediction: str,
    success: bool,
    track_steps,
    sample_idx: int,
    tokenizer,
    mask_id: int,
    max_seg_num: int,
):
    pad_id = _tokenizer_pad_id(tokenizer)
    record_steps = []

    for step_idx, step in enumerate(track_steps[:max_seg_num]):
        segments = torch.as_tensor(step["segments"])[sample_idx].long()
        segment_ids_for_decode = segments.masked_fill(segments == int(mask_id), pad_id)
        segment_texts = tokenizer.batch_decode(
            segment_ids_for_decode.tolist(),
            skip_special_tokens=True,
        )
        slot_scores = torch.as_tensor(step["slot_scores"])[sample_idx]
        chosen = int(torch.as_tensor(step["chosen_slot"])[sample_idx].item())
        chosen_for_log = chosen + 1 if 0 <= chosen < max_seg_num else -1

        segment_records = []
        for slot_idx in range(max_seg_num):
            segment_records.append(
                {
                    "slot": slot_idx + 1,
                    "text": segment_texts[slot_idx],
                    "tokens": [int(x) for x in segments[slot_idx].tolist()],
                    "score": _json_score(slot_scores[slot_idx].item()),
                }
            )

        record_steps.append(
            {
                "step": step_idx + 1,
                "chosen_slot": chosen_for_log,
                "segments": segment_records,
            }
        )

    return {
        "index": int(index),
        "prompt": _decode_prompt(tokenizer, input_ids, int(mask_id)),
        "answer": str(answer),
        "prediction": str(prediction),
        "success": bool(success),
        "decoding_order": _decoding_order_from_trace(
            track_steps,
            sample_idx,
            max_seg_num,
        ),
        "steps": record_steps,
    }

def _write_lpmdm_trace_records(
    trace_file,
    *,
    tokenizer,
    batch_X,
    batch_answers,
    samples,
    successes,
    track_steps,
    start_index: int,
    mask_id: int,
):
    max_seg_num = len(track_steps)
    for sample_idx, (answer, prediction) in enumerate(zip(batch_answers, samples)):
        record = _lpmdm_trace_record(
            index=start_index + sample_idx,
            input_ids=batch_X[sample_idx],
            answer=answer,
            prediction=prediction,
            success=bool(successes[sample_idx].item()),
            track_steps=track_steps,
            sample_idx=sample_idx,
            tokenizer=tokenizer,
            mask_id=mask_id,
            max_seg_num=max_seg_num,
        )
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_file.flush()

def _rank_trace_path(trace_path: str, rank: int):
    return f"{trace_path}.rank{rank}.tmp"

def _open_rank_trace_file(cfg, rank: int):
    trace_path = cfg.validation.get("trace_path", "eval_traces/lpmdm_eval_trace.txt")
    if trace_path is None:
        raise ValueError("validation.trace_path must be set when sampling.track=true.")
    trace_path = os.fspath(trace_path)
    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    rank_path = _rank_trace_path(trace_path, rank)
    return trace_path, rank_path, open(rank_path, "w", encoding="utf-8")

def _finalize_trace_files(trace_path: str, rank: int, world_size: int):
    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    if ddp:
        dist.barrier()

    if rank == 0:
        tmp_path = f"{trace_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as out:
            for merge_rank in range(world_size):
                rank_path = _rank_trace_path(trace_path, merge_rank)
                with open(rank_path, "r", encoding="utf-8") as inp:
                    for line in inp:
                        out.write(line)
        os.replace(tmp_path, trace_path)

        for merge_rank in range(world_size):
            rank_path = _rank_trace_path(trace_path, merge_rank)
            if os.path.exists(rank_path):
                os.remove(rank_path)

    if ddp:
        dist.barrier()

def _analysis_rank_path(plot_path: str, rank: int):
    return f"{plot_path}.rank{rank}.analysis.pt"

def _optional_path(path_value):
    if path_value is None:
        return None
    if isinstance(path_value, str) and path_value.lower() in ("", "none", "null"):
        return None
    return os.fspath(path_value)

def _validation_plot_path(cfg, confidence):
    plot_path = _optional_path(cfg.validation.get("plot_path", None))
    if plot_path is None:
        raise ValueError("validation.plot_path must be set when validation.analyze_trace=true.")

    confidence_cfg = cfg.validation.sampling.get("confidence", None)
    if _is_sequence_config(confidence_cfg) and len(list(confidence_cfg)) > 1:
        plot_root, plot_ext = os.path.splitext(plot_path)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(confidence))
        plot_path = f"{plot_root}_{slug}{plot_ext or '.png'}"

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    return plot_path

def _validation_plot_y(cfg):
    plot_y = str(cfg.validation.get("plot_y", "entropy")).strip().lower()
    if plot_y not in {"entropy", "max_prob"}:
        raise ValueError(
            "validation.plot_y must be one of: entropy, max_prob; "
            f"got {plot_y!r}."
        )
    return plot_y

def _validation_log_path(cfg):
    log_path = _optional_path(cfg.validation.get("log_path", None))
    if log_path is None:
        return None
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return log_path

def _validation_eval_itr(cfg):
    itr = cfg.validation.get("eval_itr", cfg.validation.get("itr", None))
    if isinstance(itr, str) and itr.lower() in ("", "none", "null"):
        return None
    if itr is None:
        return None
    return int(itr)

def _jsonable_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        value = value.item()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    return value

def _gather_passed_indices(local_passed_indices, world_size: int):
    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    if ddp:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_passed_indices)
    else:
        gathered = [local_passed_indices]

    merged = []
    for metric_idx in range(len(local_passed_indices)):
        metric_indices = []
        for rank_indices in gathered:
            metric_indices.extend(int(index) for index in rank_indices[metric_idx])
        merged.append(sorted(set(metric_indices)))
    return merged

def _append_passed_index_records(
    log_path: str,
    *,
    itr,
    confidence,
    temperature,
    slot_temperature,
    passed_indices_by_idx,
    total: int,
):
    cumulative_indices = set()
    with open(log_path, "a", encoding="utf-8") as log_file:
        for pass_at_k_idx, passed_indices in enumerate(passed_indices_by_idx):
            passed_indices = [int(index) for index in passed_indices]
            cumulative_indices.update(passed_indices)
            record = {
                "itr": itr,
                "confidence": _jsonable_scalar(confidence),
                "temperature": _jsonable_scalar(temperature),
                "slot_temperature": _jsonable_scalar(slot_temperature),
                "pass_at_k_idx": int(pass_at_k_idx),
                "passed_indices": passed_indices,
                "num_passed": len(passed_indices),
                "cumulative_num_passed": len(cumulative_indices),
                "total": int(total),
                "cumulative_pass_rate": len(cumulative_indices) / max(int(total), 1),
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        log_file.flush()

def _build_lpmdm_analysis_records(
    *,
    start_index: int,
    analysis_steps,
    eventual_success: torch.Tensor,
):
    if not analysis_steps:
        return []

    batch_size = int(torch.as_tensor(analysis_steps[0]["has_valid_positions"]).shape[0])
    out = []
    for sample_idx in range(batch_size):
        step_records = []
        for step in analysis_steps:
            step_records.append(
                {
                    "mass": torch.as_tensor(step["mass"])[sample_idx].clone(),
                    "entropy": torch.as_tensor(step["entropy"])[sample_idx].clone(),
                    "max_prob": torch.as_tensor(step["max_prob"])[sample_idx].clone(),
                    "highlight_token_ids": [
                        int(x) for x in step["highlight_token_ids"][sample_idx]
                    ],
                    "has_valid_positions": bool(
                        torch.as_tensor(step["has_valid_positions"])[sample_idx].item()
                    ),
                }
            )
        out.append(
            {
                "index": int(start_index + sample_idx),
                "success": bool(eventual_success[sample_idx].item()),
                "steps": step_records,
            }
        )
    return out

def _render_lpmdm_analysis_plot(
    records,
    out_path: str,
    max_steps: int,
    plot_y: str = "entropy",
):
    mpl_config_dir = os.path.join("/tmp", "matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=200, sharex=True, sharey=True)
    axes = axes.reshape(-1)
    y_key = "max_prob" if plot_y == "max_prob" else "entropy"
    y_label = (
        "Max probability / mass"
        if plot_y == "max_prob"
        else "Positional entropy"
    )

    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for record in records:
        for step in record.get("steps", [])[:max_steps]:
            if not step.get("has_valid_positions", False):
                continue
            mass = torch.as_tensor(step["mass"], dtype=torch.float32)
            y_value = torch.as_tensor(step[y_key], dtype=torch.float32)
            positive = mass > 0
            if not positive.any():
                continue
            step_mass = mass[positive]
            step_y = y_value[positive]
            x_min = min(x_min, float(step_mass.min().item()))
            x_max = max(x_max, float(step_mass.max().item()))
            y_min = min(y_min, float(step_y.min().item()))
            y_max = max(y_max, float(step_y.max().item()))

    if not math.isfinite(x_min):
        x_min, x_max = 0.0, 1.0
    if not math.isfinite(y_min):
        y_min, y_max = 0.0, 1.0
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    x_pad = max((x_max - x_min) * 0.03, 1e-8)
    y_pad = max((y_max - y_min) * 0.03, 1e-8)

    for step_idx in range(max_steps):
        ax = axes[step_idx]
        for record in records:
            steps = record.get("steps", [])
            if step_idx >= len(steps):
                continue
            step = steps[step_idx]
            if not step.get("has_valid_positions", False):
                continue

            mass = torch.as_tensor(step["mass"], dtype=torch.float32)
            y_value = torch.as_tensor(step[y_key], dtype=torch.float32)
            positive = mass > 0
            if not positive.any():
                continue

            ax.scatter(
                mass[positive].numpy(),
                y_value[positive].numpy(),
                s=6,
                c="#6b7280",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

            token_ids = torch.as_tensor(
                step.get("highlight_token_ids", []),
                dtype=torch.long,
            )
            if token_ids.numel() > 0:
                token_ids = token_ids[(token_ids >= 0) & (token_ids < mass.numel())]
                if token_ids.numel() > 0:
                    highlight_color = "#dc2626" if record.get("success", False) else "#2563eb"
                    ax.scatter(
                        mass[token_ids].numpy(),
                        y_value[token_ids].numpy(),
                        s=16,
                        c=highlight_color,
                        alpha=0.98,
                        linewidths=0,
                        rasterized=True,
                        zorder=3,
                    )

        ax.set_title(f"Step {step_idx + 1}")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    for step_idx in range(max_steps, len(axes)):
        axes[step_idx].axis("off")

    fig.supxlabel("Positional mass")
    fig.supylabel(y_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _finalize_analysis_plot(
    plot_path: str,
    rank: int,
    world_size: int,
    max_steps: int,
    plot_y: str = "entropy",
):
    ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    if ddp:
        dist.barrier()

    if rank == 0:
        merged_records = []
        for merge_rank in range(world_size):
            rank_path = _analysis_rank_path(plot_path, merge_rank)
            if os.path.exists(rank_path):
                merged_records.extend(torch.load(rank_path, map_location="cpu"))
        merged_records.sort(key=lambda record: record.get("index", -1))
        _render_lpmdm_analysis_plot(
            merged_records,
            plot_path,
            max_steps=max_steps,
            plot_y=plot_y,
        )

        for merge_rank in range(world_size):
            rank_path = _analysis_rank_path(plot_path, merge_rank)
            if os.path.exists(rank_path):
                os.remove(rank_path)

    if ddp:
        dist.barrier()

_ANS_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
def extract_gsm8k_final_answer(ans_text: str) -> str:
    """
    GSM8K 'answer' field includes reasoning and ends with: '#### 72'
    Returns the numeric string ('72', '-3', '1,234', '10.5', etc.)
    """
    m = _ANS_RE.search(ans_text)
    if not m:
        # fallback: try last number in string
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", ans_text)
        return nums[-1].replace(",", "") if nums else ""
    return m.group(1).replace(",", "")

def test_gsm8k_tokenization(mask_id: int, tokenizer_name: str | None = None):
    """
    Creates/loads:
      data/gsm8k_test/test_mdm_<tokenizer>_mask<mask_id>.json
    Format:
        input_ids: [question_ids] [SEP] [MASK] ...
        answer: numeric string
    Returns:
        X: np.ndarray[num_test, 512]
        answers: list[num_test]
    """
    tokenizer_name = tokenizer_name or TOKENIZER_NAME
    out_path = _gsm8k_cache_path(tokenizer_name, mask_id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def load_cached():
        with open(out_path, "r") as f:
            records = json.load(f)
        X = np.array([r["input_ids"] for r in records], dtype=np.int64)
        answers = [r["answer"] for r in records]
        return X, answers

    if os.path.exists(out_path):
        return load_cached()

    # ---- DDP-safe build: only rank0 writes ----
    ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if ddp else 0

    if ddp and rank != 0:
        dist.barrier()
        return load_cached()

    tokenizer = get_tokenizer(tokenizer_name)
    sep_ids = get_sep_ids(tokenizer_name)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    records = []

    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = ex.get("answer") or ""

        gold = extract_gsm8k_final_answer(a)

        q_ids = tokenizer(q, add_special_tokens=False).input_ids
        prompt_ids = q_ids + sep_ids

        # Ensure we always leave at least 1 token for mask region
        if len(prompt_ids) >= MAX_LEN:
            prompt_ids = prompt_ids[: MAX_LEN - 1]

        ids = prompt_ids + [mask_id] * (MAX_LEN - len(prompt_ids))

        records.append({
            "input_ids": ids,
            "answer": gold
        })

    # Atomic-ish write to avoid partial files
    tmp_path = out_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(records, f)
    os.replace(tmp_path, out_path)

    return load_cached()


def _count_sampling_flops(sample_fn):
    """Run ``sample_fn()`` under ``FlopCounterMode`` and return ``(output, total_flops)``.

    Hardware-agnostic: counts every matmul/SDPA that actually dispatches, so the total
    includes all candidate/slot decodes and the planner's per-iteration re-encode.
    """
    from torch.utils.flop_counter import FlopCounterMode

    fc = FlopCounterMode(display=False)
    with fc:
        out = sample_fn()
    total = sum(fc.get_flop_counts().get("Global", {}).values())
    return out, int(total)


def evaluate_ddp_gsm8k(model, cfg, device, rank: int, world_size: int, sampling):
    mask_id = cfg.data.mask_id
    track_enabled = bool(getattr(sampling, "track", False))
    analyze_enabled = bool(cfg.validation.get("analyze_trace", False))
    compute_flops = bool(cfg.validation.get("compute_flops", False))
    if track_enabled and cfg.training.strategy != "lpmdm":
        raise ValueError("validation.sampling.track=true is currently supported only for LP-MDM evaluation.")
    if analyze_enabled and cfg.training.strategy != "lpmdm":
        raise ValueError("validation.analyze_trace=true is currently supported only for LP-MDM evaluation.")
    if compute_flops and cfg.training.strategy != "lpmdm":
        raise ValueError("validation.compute_flops=true is currently supported only for LP-MDM evaluation.")

    explicit_pass_at_k = hasattr(sampling, "pass_at_k")
    pass_at_k = _as_list(getattr(sampling, "pass_at_k", 1))
    pass_at_k = [int(k) for k in pass_at_k]
    if len(pass_at_k) == 0:
        raise ValueError("sampling.pass_at_k must contain at least one value.")
    if any(k <= 0 for k in pass_at_k):
        raise ValueError(f"sampling.pass_at_k values must be positive, got {pass_at_k}.")
    max_pass_at_k = max(pass_at_k)

    # Pre-tokenize the GSM8K test set with the same tokenizer/mask_id as this run.
    tokenizer_name = _tokenizer_name_from_cfg(cfg)
    X, answers = test_gsm8k_tokenization(mask_id, tokenizer_name=tokenizer_name)
    limit = _validation_limit(cfg)
    if limit is not None:
        X = X[:limit]
        answers = answers[:limit]
    N_val = len(X)
    if N_val == 0:
        raise ValueError("No GSM8K evaluation records selected.")

    # distribute test cases
    per_rank = math.ceil(N_val / world_size)
    start = rank * per_rank
    end = min(start + per_rank, N_val)

    batch_size = int(cfg.validation.get("eval_batch_size", 128))
    num_batches = math.ceil((end - start) / batch_size)
    local_correct = torch.zeros(len(pass_at_k), dtype=torch.long)
    local_total = 0
    # Wall-clock inference timing (generation only; excludes decode + answer-checking).
    local_infer_time = 0.0
    local_infer_generations = 0
    # Hardware-agnostic inference cost (FLOPs), accumulated over all attempts/problems.
    local_infer_flops = 0
    log_path = _validation_log_path(cfg)
    local_passed_indices = [[] for _ in range(max_pass_at_k)] if log_path is not None else None
    local_analysis_records = []
    plot_path = None
    plot_y = "entropy"
    max_seg_num = int(getattr(sampling, "max_seg_num", 16))
    if analyze_enabled:
        plot_path = _validation_plot_path(cfg, getattr(sampling, "confidence", None))
        plot_y = _validation_plot_y(cfg)

    tokenizer = get_tokenizer(tokenizer_name)
    trace_path = None
    trace_file = None
    if track_enabled:
        trace_path, _, trace_file = _open_rank_trace_file(cfg, rank)

    try:
        with torch.no_grad():
            for j in tqdm(range(num_batches), desc = "Evaluating"):
                s = start + j * batch_size
                e = min(s + batch_size, end)
                batch_X_np = X[s:e]
                batch_X = torch.from_numpy(batch_X_np).long().to(device)
                batch_answers = answers[s:e]
                batch_success = torch.zeros((e - s, max_pass_at_k), dtype=torch.bool)
                batch_analysis_steps = None

                for attempt_idx in range(max_pass_at_k):
                    trace_this_attempt = track_enabled and attempt_idx == 0
                    analyze_this_attempt = analyze_enabled and attempt_idx == 0
                    trace_steps = None
                    analysis_steps = None

                    # Time the generation call only (synchronize so async CUDA work is captured).
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    infer_t0 = time.perf_counter()

                    # also support the block diffusion training
                    if cfg.training.strategy == "block":
                        block_size = cfg.training.block_size
                        samples_tensor = mdm_sampling_block(model, batch_X, block_size, mask_id, sampling, device)
                    elif cfg.training.strategy == "arm":
                        samples_tensor = arm_sampling(model, batch_X, mask_id, sampling, device)
                    elif cfg.training.strategy == "lpmdm":
                        if trace_this_attempt or analyze_this_attempt:
                            sample_out = lpmdm_sampling(
                                model,
                                batch_X,
                                mask_id,
                                sampling,
                                device,
                                track=trace_this_attempt,
                                analysis=analyze_this_attempt,
                            )
                            if trace_this_attempt and analyze_this_attempt:
                                samples_tensor, trace_steps, analysis_steps = sample_out
                            elif trace_this_attempt:
                                samples_tensor, trace_steps = sample_out
                            else:
                                samples_tensor, analysis_steps = sample_out
                        elif compute_flops:
                            samples_tensor, attempt_flops = _count_sampling_flops(
                                lambda: lpmdm_sampling(model, batch_X, mask_id, sampling, device)
                            )
                            local_infer_flops += attempt_flops
                        else:
                            samples_tensor = lpmdm_sampling(model, batch_X, mask_id, sampling, device)
                    else:
                        samples_tensor = mdm_sampling(model, batch_X, mask_id, sampling, device, arm_init=cfg.model.arm_init!="none")

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    local_infer_time += time.perf_counter() - infer_t0
                    local_infer_generations += (e - s)

                    # tokenizer by default doesn't have mask_id
                    samples_tensor = samples_tensor.masked_fill(samples_tensor == mask_id, _tokenizer_pad_id(tokenizer))

                    # sample preproceessing, and extract the answer part
                    sample_ids = samples_tensor.cpu().numpy()
                    samples = tokenizer.batch_decode(sample_ids, skip_special_tokens=True)
                
                    for sample_idx, (sample, answer) in enumerate(zip(samples, batch_answers)):
                        batch_success[sample_idx, attempt_idx] = evaluate_samples(sample, answer)

                    if analyze_this_attempt:
                        batch_analysis_steps = analysis_steps
                    if trace_this_attempt:
                        _write_lpmdm_trace_records(
                            trace_file,
                            tokenizer=tokenizer,
                            batch_X=batch_X_np,
                            batch_answers=batch_answers,
                            samples=samples,
                            successes=batch_success[:, attempt_idx],
                            track_steps=trace_steps,
                            start_index=s,
                            mask_id=mask_id,
                        )

                if analyze_enabled and batch_analysis_steps is not None:
                    local_analysis_records.extend(
                        _build_lpmdm_analysis_records(
                            start_index=s,
                            analysis_steps=batch_analysis_steps,
                            eventual_success=batch_success.any(dim=1),
                        )
                    )
                if local_passed_indices is not None:
                    for attempt_idx in range(max_pass_at_k):
                        passed_rows = torch.nonzero(
                            batch_success[:, attempt_idx],
                            as_tuple=False,
                        ).view(-1).tolist()
                        local_passed_indices[attempt_idx].extend(
                            int(s + row_idx) for row_idx in passed_rows
                        )

                for metric_idx, k in enumerate(pass_at_k):
                    passed_mask = batch_success[:, :k].any(dim=1)
                    local_correct[metric_idx] += passed_mask.sum()
                local_total += e - s
    finally:
        if trace_file is not None:
            trace_file.close()
    
    # accumulate succcess rates
    tensor = torch.cat([local_correct, torch.tensor([local_total], dtype=torch.long)]).to(device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    values = tensor.tolist()
    global_correct = values[:-1]
    global_total = values[-1]
    results = {
        f"pass_at_{k}": correct / global_total
        for k, correct in zip(pass_at_k, global_correct)
    }

    # Aggregate inference wall-clock across ranks and report per-sample latency.
    time_tensor = torch.tensor(
        [local_infer_time, float(local_infer_generations), float(local_infer_flops)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)
    total_infer_time, total_generations, total_infer_flops = time_tensor.tolist()
    # Hardware-agnostic per-problem inference cost (summed over all pass@k attempts).
    flops_per_problem = total_infer_flops / global_total if global_total > 0 else float("nan")
    if compute_flops:
        results["avg_flops_per_problem"] = flops_per_problem
    # Each problem is generated max_pass_at_k times; normalize to per-(problem) cost too.
    sec_per_generation = total_infer_time / total_generations if total_generations > 0 else float("nan")
    sec_per_sample = sec_per_generation * max_pass_at_k
    if rank == 0:
        print(
            f"[inference timing] total generation time: {total_infer_time:.2f}s over "
            f"{int(total_generations)} generations across {world_size} rank(s) "
            f"(pass@k attempts={max_pass_at_k})"
        )
        print(
            f"[inference timing] per-generation: {sec_per_generation * 1e3:.1f} ms | "
            f"per-sample (all {max_pass_at_k} attempts): {sec_per_sample * 1e3:.1f} ms "
            f"({sec_per_sample:.3f} s/sample)"
        )
        if compute_flops:
            if flops_per_problem >= 1e12:
                flops_str = f"{flops_per_problem / 1e12:.3f} TFLOPs"
            else:
                flops_str = f"{flops_per_problem / 1e9:.3f} GFLOPs"
            print(
                f"[inference flops] avg per-problem total inference FLOPs: {flops_str} "
                f"(sum over {max_pass_at_k} attempt(s); "
                f"{total_infer_flops / 1e9:.1f} GFLOPs over {int(global_total)} problems)"
            )
    if track_enabled:
        _finalize_trace_files(trace_path, rank, world_size)
    if analyze_enabled:
        torch.save(local_analysis_records, _analysis_rank_path(plot_path, rank))
        _finalize_analysis_plot(
            plot_path,
            rank,
            world_size,
            max_steps=max_seg_num,
            plot_y=plot_y,
        )
    if local_passed_indices is not None:
        merged_passed_indices = _gather_passed_indices(local_passed_indices, world_size)
        if rank == 0:
            _append_passed_index_records(
                log_path,
                itr=_validation_eval_itr(cfg),
                confidence=getattr(sampling, "confidence", None),
                temperature=getattr(sampling, "temperature", None),
                slot_temperature=getattr(sampling, "slot_temperature", None),
                passed_indices_by_idx=merged_passed_indices,
                total=global_total,
            )
    if not explicit_pass_at_k and pass_at_k == [1]:
        return results["pass_at_1"]
    return results


def evaluate_samples(sample: str, answer: str, timeout_s: float = 1.0) -> bool:
    """
    sample: model output (string)
    answer: GSM8K answer (string)

    Key fix:
      - timeout now covers BOTH exec(code) and fn() execution
      - prevents a single pathological sample from hanging a rank forever
    """
    code = _extract_code(sample)

    try:
        with _time_limit(timeout_s):
            ns = _safe_exec_no_timer(code)

            fn = ns.get("simple_math_problem", None)
            if fn is None:
                return False

            out = fn()  # now time-bounded

    except (_Timeout, Exception):
        return False

    pred = _to_number(out)
    gold = _to_number(answer)
    return _numbers_equal(pred, gold)

# -----------------------------
# Code execution functions
# -----------------------------
class _Timeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _Timeout()


@contextlib.contextmanager
def _time_limit(timeout_s: float):
    """
    Hard wall-clock time limit using SIGALRM/ITIMER_REAL (POSIX).
    Note: works only in the main thread of the process.
    """
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    old_handler = None
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        if has_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)


def _safe_exec_no_timer(code: str):
    """
    Executes code in a restricted environment (no timeout here).
    Timeout should be applied by wrapping the whole evaluate step with _time_limit().
    """
    import math as _math

    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "enumerate": enumerate,
        "int": int, "float": float, "str": str, "bool": bool,
        "round": round,
        "print": print,
    }

    def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math":
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import blocked: {name}")

    safe_builtins["__import__"] = _limited_import

    ns = {
        "__builtins__": safe_builtins,
        "math": _math,
    }

    # Reduce noisy compile-time warnings from weird generated code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(code, ns, ns)

    return ns

def _extract_code(text: str) -> str:
    """
    Heuristics:
      - If code fences exist, prefer fenced block.
      - Else, start from first 'def ' if present.
      - Strip special tokens.
      - Trim trailing garbage until it compiles (best-effort).
    """
    # Cut at common special tokens
    for stopper in ["<|endoftext|>", "<|eot_id|>", "</s>"]:
        if stopper in text:
            text = text.split(stopper, 1)[0]

    # Prefer fenced code
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)

    # If it contains 'def', slice from first def
    i = text.find("def ")
    if i != -1:
        text = text[i:]

    text = text.strip()

    # Best-effort trimming to make it syntactically valid
    # (Useful if sampling adds junk after valid Python)
    lines = text.splitlines()
    for k in range(0, min(50, len(lines))):
        candidate = "\n".join(lines[: len(lines) - k]).strip()
        if not candidate:
            continue
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except SyntaxError:
            continue

    return text


# -----------------------------
# Numeric handling functions
# -----------------------------

def _numbers_equal(pred, gold):
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) or isinstance(gold, float):
        return abs(float(pred) - float(gold)) <= 1e-3
    return int(pred) == int(gold)

def _to_number(x):
    """
    Normalize return values to int or float where possible.
    """
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        xf = float(x)
        if abs(xf - round(xf)) < 1e-6:
            return int(round(xf))
        return xf
    if isinstance(x, str):
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", x)
        if not m:
            return None
        s = m.group(0).replace(",", "")
        if s.count(".") == 1:
            f = float(s)
            if abs(f - round(f)) < 1e-6:
                return int(round(f))
            return f
        return int(s)
    # tuples/lists etc -> not supported for GSM8K scoring
    return None


if __name__ == "__main__":
    # tokenize the GSM8K test set first
    # test_gsm8k_tokenization(MASK_ID)

    # sanity check the eval loop with one tinygsm example
    ds = load_dataset("TinyGSM/TinyGSM", split = "train")
    ex = ds[0]

    q, a  = ex["question"], ex["code"]

    ns = _safe_exec( _extract_code(q + "\n" + a) , timeout_s = 1.0)
    out = ns["simple_math_problem"]()
    gold = str(_to_number(out))

    ok = evaluate_samples( a , gold)

    print("Sanity check passed: ", ok)
    print("--------------------------------")
    print("Question: ", q)
    print("--------------------------------")
    print("Code: ", a)
    print("--------------------------------")
    print("Answer: ", gold)

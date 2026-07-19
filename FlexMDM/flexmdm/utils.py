"""Shared FlexMDM evaluation utilities."""

from __future__ import annotations

import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .architecture import build_flexmdm_model


ANSI_RESET = "\033[0m"
ANSI_XBLUE = "\033[38;2;65;105;225m"
ANSI_XXGREEN = "\033[38;2;0;159;134m"


def use_color(enabled: bool = True) -> bool:
    return bool(enabled) and os.environ.get("NO_COLOR") is None


def colorize(text: str, color: str, *, enabled: bool = True) -> str:
    if not use_color(enabled):
        return text
    return f"{color}{text}{ANSI_RESET}"


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return {}
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("PyYAML is required to read inference config files.") from exc
    with open(expanded, "r") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} did not load as a mapping.")
    return loaded


def config_get(
    config: Mapping[str, Any],
    *path: str,
    default: Any = None,
) -> Any:
    cur: Any = config
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name!r}")
    return dtype


def report_weight_health(model: torch.nn.Module) -> None:
    """Print whether any parameter has NaN/Inf."""
    bad: list[tuple[str, str]] = []
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            n_nan = int(torch.isnan(param).sum().item())
            n_inf = int(torch.isinf(param).sum().item())
            bad.append((name, f"nan={n_nan} inf={n_inf}"))
    if bad:
        print("[weight check] non-finite params:")
        for name, info in bad[:20]:
            print(f"  {name}: {info}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
    else:
        print("[weight check] all params finite.")


def load_model_and_tokenizer(
    *,
    checkpoint_dir: str,
    max_length: int,
    torch_dtype_name: str = "bfloat16",
    attn_implementation: str | None = "sdpa",
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a FlexMDM checkpoint (Dream backbone + flexmdm_extras.pt).

    ``attn_implementation`` defaults to ``"sdpa"`` (works everywhere). Pass
    ``"flash_attention_2"`` for speed (requires flash-attn), or ``"eager"``
    to bit-match the released evaluation traces (see evals/REPRODUCIBILITY.md).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, trust_remote_code=trust_remote_code
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id.")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(torch_dtype_name)
    hf_config = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=trust_remote_code
    )
    model_kwargs: dict[str, Any] = {
        "config": hf_config,
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    backbone = AutoModel.from_pretrained(checkpoint_dir, **model_kwargs)
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False

    model = build_flexmdm_model(
        backbone, tokenizer=tokenizer, max_length=max_length
    )
    model.to(dtype=dtype)

    extras_path = os.path.join(checkpoint_dir, "flexmdm_extras.pt")
    if not os.path.isfile(extras_path):
        raise FileNotFoundError(f"flexmdm_extras.pt not found at {extras_path}")
    extras = torch.load(extras_path, map_location="cpu", weights_only=True)
    incompat = model.load_state_dict(extras, strict=False)
    expected_extras_prefixes = ("insertion_head.", "time_mlp.", "temb_mods.")
    missing_extras = [
        key
        for key in incompat.missing_keys
        if key.startswith(expected_extras_prefixes)
    ]
    if missing_extras:
        print(f"[load] WARNING: missing FlexMDM extras keys: {missing_extras}")
    if incompat.unexpected_keys:
        print(f"[load] WARNING: unexpected keys in extras: {incompat.unexpected_keys}")
    print(
        f"[load] extras applied: {len(extras)} tensors. "
        f"missing(non-extras-backbone)={len(incompat.missing_keys) - len(missing_extras)}, "
        f"unexpected={len(incompat.unexpected_keys)}"
    )
    return model, tokenizer


def decode_ids_with_masks(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    mask_id: int,
    mask_text: str = "[M]",
) -> str:
    """Decode token ids while rendering tokenizer mask ids as a stable marker."""
    parts: list[str] = []
    chunk: list[int] = []

    def flush_chunk() -> None:
        if chunk:
            parts.append(tokenizer.decode(chunk, skip_special_tokens=False))
            chunk.clear()

    for token_id in token_ids:
        if int(token_id) == int(mask_id):
            flush_chunk()
            parts.append(mask_text)
        else:
            chunk.append(int(token_id))
    flush_chunk()
    return "".join(parts)


def _box_width(width: int | None = None) -> int:
    if width is not None:
        return max(40, int(width))
    terminal_width = shutil.get_terminal_size((100, 24)).columns
    return max(60, min(120, terminal_width))


def _wrap_box_lines(body: str, inner_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in body.splitlines() or [""]:
        wrapped = textwrap.wrap(
            raw_line,
            width=inner_width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines


def print_box(
    title: str,
    body: str,
    *,
    subtitle: str | None = None,
    color: bool = True,
    width: int | None = None,
) -> None:
    box_width = _box_width(width)
    inner_width = box_width - 4
    top_border = "+" + "=" * (inner_width + 2) + "+"
    mid_border = "+" + "-" * (inner_width + 2) + "+"
    bottom_border = "+" + "=" * (inner_width + 2) + "+"
    title_text = title if subtitle is None else f"{title} | {subtitle}"
    title_text = title_text[:inner_width]

    print(colorize(top_border, ANSI_XBLUE, enabled=color))
    print(
        colorize("| ", ANSI_XBLUE, enabled=color)
        + colorize(title_text.ljust(inner_width), ANSI_XXGREEN, enabled=color)
        + colorize(" |", ANSI_XBLUE, enabled=color)
    )
    print(colorize(mid_border, ANSI_XBLUE, enabled=color))
    for line in _wrap_box_lines(body, inner_width):
        print(
            colorize("| ", ANSI_XBLUE, enabled=color)
            + line.ljust(inner_width)
            + colorize(" |", ANSI_XBLUE, enabled=color)
        )
    print(colorize(bottom_border, ANSI_XBLUE, enabled=color))


def print_decoded_snapshot(
    tokenizer: Any,
    sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    step_idx: int,
    total_steps: int,
    mask_id: int,
    prompt_mask: torch.Tensor | None = None,
    sample_idx: int | None = None,
    mask_text: str = "[M]",
    color: bool = True,
) -> None:
    active = attention_mask.bool()
    if prompt_mask is not None:
        active = active & ~prompt_mask.bool()
    ids = sequence[active].tolist()
    text = decode_ids_with_masks(
        tokenizer,
        ids,
        mask_id=mask_id,
        mask_text=mask_text,
    )
    subtitle_parts = [f"step {step_idx}/{total_steps}"]
    if sample_idx is not None:
        subtitle_parts.append(f"sample {sample_idx}")
    subtitle_parts.append(f"answer length {int(active.sum().item())}")
    subtitle = " | ".join(subtitle_parts)
    print_box("FlexMDM answer sample", text, subtitle=subtitle, color=color)


def print_decoded_batch_snapshot(
    tokenizer: Any,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    step_idx: int,
    total_steps: int,
    mask_id: int,
    prompt_mask: torch.Tensor | None = None,
    mask_text: str = "[M]",
    color: bool = True,
    max_samples: int | None = None,
) -> None:
    batch_size = int(sequences.shape[0])
    limit = batch_size if max_samples is None else min(batch_size, int(max_samples))
    for sample_idx in range(limit):
        sample_prompt_mask = None
        if prompt_mask is not None:
            sample_prompt_mask = prompt_mask[sample_idx]
        print_decoded_snapshot(
            tokenizer,
            sequences[sample_idx],
            attention_mask[sample_idx],
            step_idx=step_idx,
            total_steps=total_steps,
            mask_id=mask_id,
            prompt_mask=sample_prompt_mask,
            sample_idx=sample_idx if batch_size > 1 else None,
            mask_text=mask_text,
            color=color,
        )


def print_trace_snapshots(
    tokenizer: Any,
    history: Sequence[torch.Tensor],
    attention_history: Sequence[torch.Tensor],
    *,
    trace_every: int,
    mask_id: int,
    prompt_mask: torch.Tensor | None = None,
    mask_text: str = "[M]",
    color: bool = True,
    max_samples: int | None = None,
) -> None:
    n = len(history)
    if n == 0:
        return
    total_steps = n - 1
    keep = sorted(set(list(range(0, n, trace_every)) + [n - 1]))
    print(
        colorize(
            f"[trace] decoding {len(keep)} of {n} snapshots "
            f"(every {trace_every} steps)",
            ANSI_XXGREEN,
            enabled=color,
        )
    )
    for idx in keep:
        print_decoded_batch_snapshot(
            tokenizer,
            history[idx],
            attention_history[idx],
            step_idx=idx,
            total_steps=total_steps,
            mask_id=mask_id,
            prompt_mask=prompt_mask,
            mask_text=mask_text,
            color=color,
            max_samples=max_samples,
        )


def print_generation_result(
    *,
    prompt: str,
    answer: str,
    full: str,
    color: bool = True,
) -> None:
    print_box("Prompt", prompt, color=color)
    print_box("Answer", answer, color=color)
    print_box("Full sequence", full, color=color)


def _trace_snapshot_payload(
    tokenizer: Any,
    sequence: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    mask_id: int,
    mask_text: str,
    prediction_tokens: torch.Tensor | None = None,
    prediction_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    sequence = sequence.detach().cpu()
    attention_mask = attention_mask.detach().cpu().bool()
    prompt_mask = prompt_mask.detach().cpu().bool()
    if prediction_tokens is not None:
        prediction_tokens = prediction_tokens.detach().cpu()
    if prediction_scores is not None:
        prediction_scores = prediction_scores.detach().cpu()
    active = attention_mask
    answer_active = active & ~prompt_mask
    full_ids = sequence[active].tolist()
    answer_ids = sequence[answer_active].tolist()
    answer_tokens = sequence[answer_active]
    mask_count = int(answer_tokens.eq(mask_id).sum().item())
    answer_len = int(answer_active.sum().item())
    full_len = int(active.sum().item())
    filled = max(0, answer_len - mask_count)
    mask_predictions: list[dict[str, Any]] = []
    if (
        prediction_tokens is not None
        and prediction_scores is not None
        and prediction_tokens.numel() > 0
    ):
        masked_answer = answer_active & sequence.eq(mask_id)
        for pos_tensor in masked_answer.nonzero(as_tuple=False).flatten():
            pos = int(pos_tensor.item())
            token_row = prediction_tokens[pos]
            score_row = prediction_scores[pos]
            if token_row.ndim == 0:
                token_row = token_row.view(1)
                score_row = score_row.view(1)
            choices = []
            for rank in range(int(token_row.shape[0])):
                token_id = int(token_row[rank].item())
                prob = float(score_row[rank].item())
                choices.append(
                    {
                        "rank": rank + 1,
                        "token_id": token_id,
                        "token": decode_ids_with_masks(
                            tokenizer,
                            [token_id],
                            mask_id=mask_id,
                            mask_text=mask_text,
                        ),
                        "prob": prob,
                    }
                )
            mask_predictions.append(
                {
                    "absolute_index": pos,
                    "full_index": int(active[: pos + 1].sum().item()) - 1,
                    "answer_index": int(answer_active[: pos + 1].sum().item()) - 1,
                    "choices": choices,
                }
            )
    return {
        "step": int(step),
        "total_steps": int(total_steps),
        "progress": 0.0 if total_steps <= 0 else float(step) / float(total_steps),
        "full": decode_ids_with_masks(
            tokenizer,
            full_ids,
            mask_id=mask_id,
            mask_text=mask_text,
        ),
        "answer": decode_ids_with_masks(
            tokenizer,
            answer_ids,
            mask_id=mask_id,
            mask_text=mask_text,
        ),
        "full_len": full_len,
        "answer_len": answer_len,
        "mask_count": mask_count,
        "filled_count": filled,
        "mask_prediction_count": len(mask_predictions),
        "mask_predictions": mask_predictions,
    }


def write_trace_html(
    path: str | os.PathLike[str],
    *,
    tokenizer: Any,
    history: Sequence[torch.Tensor],
    attention_history: Sequence[torch.Tensor],
    history_steps: Sequence[int] | None,
    total_steps: int,
    prompts: Sequence[str],
    prompt_mask: torch.Tensor,
    mask_id: int,
    mask_text: str = "[M]",
    prediction_history: Sequence[torch.Tensor] | None = None,
    prediction_score_history: Sequence[torch.Tensor] | None = None,
) -> str:
    """Write a self-contained HTML viewer for FlexMDM sampling snapshots.

    Output format embeds a ``const DATA = {...}`` JSON blob with full per-step
    token state per sample, which is the format the trace-analysis tooling in
    ``evals/tree_analysis`` consumes.
    """
    if len(history) != len(attention_history):
        raise ValueError("history and attention_history must have the same length.")
    if not history:
        raise ValueError("Cannot write an HTML trace without snapshots.")
    if history_steps is None:
        history_steps = list(range(len(history)))
    if len(history_steps) != len(history):
        raise ValueError("history_steps must match history length.")
    has_prediction_history = (
        prediction_history is not None or prediction_score_history is not None
    )
    if has_prediction_history:
        if prediction_history is None or prediction_score_history is None:
            raise ValueError(
                "prediction_history and prediction_score_history must be passed together."
            )
        if len(prediction_history) != len(history):
            raise ValueError("prediction_history must match history length.")
        if len(prediction_score_history) != len(history):
            raise ValueError("prediction_score_history must match history length.")

    prompt_mask_cpu = prompt_mask.detach().cpu().bool()
    batch_size = int(history[0].shape[0])
    samples: list[dict[str, Any]] = []
    for sample_idx in range(batch_size):
        prompt = str(prompts[sample_idx]) if sample_idx < len(prompts) else ""
        snapshots = []
        for snapshot_idx, (step, sequence, attention) in enumerate(zip(
            history_steps,
            history,
            attention_history,
        )):
            prediction_tokens = None
            prediction_scores = None
            if prediction_history is not None and prediction_score_history is not None:
                prediction_tokens = prediction_history[snapshot_idx][sample_idx]
                prediction_scores = prediction_score_history[snapshot_idx][sample_idx]
            snapshots.append(
                _trace_snapshot_payload(
                    tokenizer,
                    sequence[sample_idx],
                    attention[sample_idx],
                    prompt_mask_cpu[sample_idx],
                    step=int(step),
                    total_steps=total_steps,
                    mask_id=mask_id,
                    mask_text=mask_text,
                    prediction_tokens=prediction_tokens,
                    prediction_scores=prediction_scores,
                )
            )
        samples.append(
            {
                "sample_idx": sample_idx,
                "prompt": prompt,
                "snapshots": snapshots,
            }
        )

    payload = {
        "title": "FlexMDM Sampling Trace",
        "total_steps": int(total_steps),
        "snapshot_count": len(history),
        "sample_count": batch_size,
        "mask_text": mask_text,
        "samples": samples,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FlexMDM Sampling Trace</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7deea;
      --panel: #ffffff;
      --teal: #008f83;
      --green: #2e7d32;
      --amber: #c67a00;
      --red: #c2414b;
      --blue: #3157a4;
      --shadow: 0 10px 30px rgba(30, 41, 59, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
    }}
    header {{
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.08;
      letter-spacing: 0;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(160px, 220px) auto;
      gap: 14px;
      align-items: center;
      margin-bottom: 14px;
    }}
    select, button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 7px;
      height: 34px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{
      min-width: 38px;
      cursor: pointer;
    }}
    button:hover, select:hover {{ border-color: #aeb8c9; }}
    .timeline {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
    }}
    .buttons {{ display: flex; gap: 6px; }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--teal);
    }}
    .step-label {{
      color: var(--muted);
      min-width: 122px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .sample-list {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 8px;
      max-height: 180px;
      overflow-y: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 8px;
      margin-bottom: 14px;
    }}
    .sample-option {{
      display: grid;
      gap: 3px;
      height: auto;
      min-height: 62px;
      min-width: 0;
      width: 100%;
      text-align: left;
      padding: 8px 10px;
      background: #ffffff;
    }}
    .sample-option.active {{
      border-color: var(--teal);
      background: #eefaf8;
      box-shadow: inset 3px 0 0 var(--teal);
    }}
    .sample-title {{
      font-weight: 700;
      font-size: 13px;
    }}
    .sample-detail,
    .sample-preview {{
      color: var(--muted);
      font-size: 12px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcff;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .stat strong {{
      display: block;
      margin-top: 2px;
      font-size: 22px;
      font-variant-numeric: tabular-nums;
    }}
    .prediction-panel {{
      margin-top: 14px;
    }}
    .prediction-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }}
    .prediction-meta {{
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }}
    .prediction-list {{
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
    }}
    .prediction-row {{
      display: grid;
      grid-template-columns: minmax(96px, 130px) minmax(0, 1fr);
      gap: 10px;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
    }}
    .prediction-row:last-child {{
      border-bottom: 0;
    }}
    .prediction-pos {{
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }}
    .prediction-choices {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      min-width: 0;
    }}
    .token-chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #ffffff;
      padding: 3px 6px;
      font: 12px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    .token-chip:first-child {{
      border-color: #a8d8d2;
      background: #eefaf8;
    }}
    .token-prob {{
      color: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
      font-variant-numeric: tabular-nums;
    }}
    .empty-state {{
      color: var(--muted);
      padding: 12px;
    }}
    canvas {{
      width: 100%;
      height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      display: block;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      margin: 9px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 3px;
      display: inline-block;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    pre {{
      margin: 0;
      min-height: 220px;
      max-height: 420px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1720;
      color: #e6edf3;
      padding: 14px;
      font: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    }}
    .mask {{
      color: #101418;
      background: #ffd166;
      border-radius: 4px;
      padding: 0 3px;
      margin: 0 1px;
    }}
    details {{
      margin-top: 14px;
    }}
    summary {{
      cursor: pointer;
      color: var(--blue);
      font-weight: 600;
      margin-bottom: 8px;
    }}
      @media (max-width: 820px) {{
      header, .controls, .timeline, .grid {{ grid-template-columns: 1fr; display: grid; }}
      header {{ align-items: flex-start; }}
      .meta {{ justify-content: flex-start; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .sample-list {{ grid-template-columns: 1fr; }}
      .prediction-head, .prediction-row {{ grid-template-columns: 1fr; display: grid; }}
      .step-label {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>FlexMDM Sampling Trace</h1>
      <div class="meta">
        <span class="pill" id="sampleMeta"></span>
        <span class="pill" id="snapshotMeta"></span>
        <span class="pill" id="totalMeta"></span>
      </div>
    </header>
    <section class="panel">
      <div class="controls">
        <select id="sampleSelect"></select>
        <div class="timeline">
          <div class="buttons">
            <button id="prevBtn" title="Previous snapshot">&#8592;</button>
            <button id="playBtn" title="Play or pause">&#9658;</button>
            <button id="nextBtn" title="Next snapshot">&#8594;</button>
          </div>
          <input id="stepSlider" type="range" min="0" value="0">
          <div class="step-label" id="stepLabel"></div>
        </div>
      </div>
      <div class="sample-list" id="sampleList" role="listbox" aria-label="Batch samples"></div>
      <canvas id="tracePlot"></canvas>
      <div class="legend">
        <span class="key"><span class="swatch" style="background: var(--green)"></span>answer length</span>
        <span class="key"><span class="swatch" style="background: var(--amber)"></span>mask count</span>
        <span class="key"><span class="swatch" style="background: var(--red)"></span>current step</span>
      </div>
      <div class="stats">
        <div class="stat"><span>Step</span><strong id="statStep"></strong></div>
        <div class="stat"><span>Answer Tokens</span><strong id="statAnswer"></strong></div>
        <div class="stat"><span>Filled Tokens</span><strong id="statFilled"></strong></div>
        <div class="stat"><span>Masks</span><strong id="statMasks"></strong></div>
      </div>
      <section class="prediction-panel">
        <div class="prediction-head">
          <h2>Masked Position Predictions</h2>
          <span class="prediction-meta" id="predictionMeta"></span>
        </div>
        <div class="prediction-list" id="predictionList"></div>
      </section>
      <div class="grid">
        <section>
          <h2>Answer Snapshot</h2>
          <pre id="answerText"></pre>
        </section>
        <section>
          <h2>Full Sequence</h2>
          <pre id="fullText"></pre>
        </section>
      </div>
      <details>
        <summary>Prompt</summary>
        <pre id="promptText"></pre>
      </details>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const state = {{ sample: 0, idx: 0, timer: null }};
    const maskText = DATA.mask_text || "[M]";
    const els = {{
      sampleSelect: document.getElementById("sampleSelect"),
      sampleList: document.getElementById("sampleList"),
      slider: document.getElementById("stepSlider"),
      prev: document.getElementById("prevBtn"),
      next: document.getElementById("nextBtn"),
      play: document.getElementById("playBtn"),
      stepLabel: document.getElementById("stepLabel"),
      sampleMeta: document.getElementById("sampleMeta"),
      snapshotMeta: document.getElementById("snapshotMeta"),
      totalMeta: document.getElementById("totalMeta"),
      statStep: document.getElementById("statStep"),
      statAnswer: document.getElementById("statAnswer"),
      statFilled: document.getElementById("statFilled"),
      statMasks: document.getElementById("statMasks"),
      predictionMeta: document.getElementById("predictionMeta"),
      predictionList: document.getElementById("predictionList"),
      answerText: document.getElementById("answerText"),
      fullText: document.getElementById("fullText"),
      promptText: document.getElementById("promptText"),
      plot: document.getElementById("tracePlot"),
    }};

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function renderMaskedText(el, value) {{
      const text = String(value || "");
      if (!maskText || !text.includes(maskText)) {{
        el.textContent = text;
        return;
      }}
      el.innerHTML = text
        .split(maskText)
        .map(escapeHtml)
        .join(`<span class="mask">${{escapeHtml(maskText)}}</span>`);
    }}

    function formatToken(token) {{
      const text = String(token ?? "");
      if (!text) return "[empty]";
      const quoted = JSON.stringify(text);
      return quoted.length > 24 ? `${{quoted.slice(0, 23)}}...` : quoted;
    }}

    function formatProb(prob) {{
      const value = Number(prob);
      if (!Number.isFinite(value)) return "";
      return `${{(value * 100).toFixed(1)}}%`;
    }}

    function renderPredictions(snapshot) {{
      const predictions = snapshot.mask_predictions || [];
      els.predictionMeta.textContent =
        `${{predictions.length}} masked position${{predictions.length === 1 ? "" : "s"}}`;
      els.predictionList.innerHTML = "";
      if (!predictions.length) {{
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No masked answer positions at this snapshot.";
        els.predictionList.appendChild(empty);
        return;
      }}
      predictions.forEach(prediction => {{
        const row = document.createElement("div");
        row.className = "prediction-row";

        const pos = document.createElement("div");
        pos.className = "prediction-pos";
        pos.textContent =
          `answer ${{Number(prediction.answer_index) + 1}} / full ${{Number(prediction.full_index) + 1}}`;

        const choices = document.createElement("div");
        choices.className = "prediction-choices";
        (prediction.choices || []).forEach(choice => {{
          const chip = document.createElement("span");
          chip.className = "token-chip";

          const token = document.createElement("span");
          token.textContent = `${{choice.rank}}: ${{formatToken(choice.token)}}`;

          const prob = document.createElement("span");
          prob.className = "token-prob";
          prob.textContent = formatProb(choice.prob);

          chip.append(token, prob);
          choices.appendChild(chip);
        }});

        row.append(pos, choices);
        els.predictionList.appendChild(row);
      }});
    }}

    function currentSample() {{
      return DATA.samples[state.sample];
    }}

    function currentSnapshots() {{
      return currentSample().snapshots;
    }}

    function currentSnapshot() {{
      return currentSnapshots()[state.idx];
    }}

    function sampleSummary(sample) {{
      const snaps = sample.snapshots || [];
      const last = snaps[snaps.length - 1] || {{}};
      const answerLen = Number(last.answer_len || 0);
      const maskCount = Number(last.mask_count || 0);
      const filledCount = Number(last.filled_count || 0);
      return `${{answerLen}} tokens, ${{filledCount}} filled, ${{maskCount}} masks`;
    }}

    function samplePreview(sample) {{
      const snaps = sample.snapshots || [];
      const last = snaps[snaps.length - 1] || {{}};
      const answerLine = String(last.answer || "").split("\\n").find(line => line.trim()) || "";
      if (answerLine.trim()) return answerLine.trim();
      const prompt = String(sample.prompt || "");
      const firstLine = prompt.split("\\n").find(line => line.trim()) || "";
      return firstLine.trim();
    }}

    function updateSampleList(scrollActive = false) {{
      els.sampleList.querySelectorAll(".sample-option").forEach((button, idx) => {{
        const active = idx === state.sample;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
        if (active && scrollActive) {{
          button.scrollIntoView({{ block: "nearest", inline: "nearest" }});
        }}
      }});
    }}

    function selectSample(index) {{
      state.sample = Math.max(0, Math.min(Number(index), DATA.samples.length - 1));
      state.idx = 0;
      els.sampleSelect.value = String(state.sample);
      setPlaying(false);
      render();
      updateSampleList(true);
    }}

    function drawPlot() {{
      const canvas = els.plot;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * dpr));
      canvas.height = Math.max(180, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const pad = {{ left: 42, right: 18, top: 18, bottom: 30 }};
      const innerW = Math.max(1, w - pad.left - pad.right);
      const innerH = Math.max(1, h - pad.top - pad.bottom);
      const snaps = currentSnapshots();
      const maxY = Math.max(1, ...snaps.map(s => Math.max(s.answer_len, s.mask_count)));
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#e1e7f0";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#667085";
      ctx.font = "12px ui-sans-serif, system-ui";
      for (let i = 0; i <= 4; i++) {{
        const y = pad.top + innerH * i / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();
        const label = Math.round(maxY * (1 - i / 4));
        ctx.fillText(String(label), 8, y + 4);
      }}
      function point(snapshot, value) {{
        const x = pad.left + innerW * (snapshot.step / Math.max(1, DATA.total_steps));
        const y = pad.top + innerH * (1 - value / maxY);
        return [x, y];
      }}
      function line(metric, color) {{
        ctx.beginPath();
        snaps.forEach((s, i) => {{
          const [x, y] = point(s, s[metric]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.stroke();
      }}
      line("answer_len", "#2e7d32");
      line("mask_count", "#c67a00");
      const snap = currentSnapshot();
      const cursorX = pad.left + innerW * (snap.step / Math.max(1, DATA.total_steps));
      ctx.strokeStyle = "#c2414b";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(cursorX, pad.top);
      ctx.lineTo(cursorX, h - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = "#667085";
      ctx.fillText("0", pad.left - 3, h - 8);
      ctx.textAlign = "right";
      ctx.fillText(String(DATA.total_steps), w - pad.right, h - 8);
      ctx.textAlign = "left";
    }}

    function render() {{
      const snaps = currentSnapshots();
      state.idx = Math.max(0, Math.min(state.idx, snaps.length - 1));
      const snap = currentSnapshot();
      els.slider.max = String(snaps.length - 1);
      els.slider.value = String(state.idx);
      els.stepLabel.textContent = `step ${{snap.step}} / ${{DATA.total_steps}}`;
      els.statStep.textContent = snap.step;
      els.statAnswer.textContent = snap.answer_len;
      els.statFilled.textContent = snap.filled_count;
      els.statMasks.textContent = snap.mask_count;
      renderMaskedText(els.answerText, snap.answer);
      renderMaskedText(els.fullText, snap.full);
      els.promptText.textContent = currentSample().prompt || "";
      renderPredictions(snap);
      updateSampleList();
      drawPlot();
    }}

    function setPlaying(isPlaying) {{
      if (state.timer) {{
        clearInterval(state.timer);
        state.timer = null;
      }}
      els.play.textContent = isPlaying ? "\\u275A\\u275A" : "\\u25B6";
      if (!isPlaying) return;
      state.timer = setInterval(() => {{
        const snaps = currentSnapshots();
        state.idx = (state.idx + 1) % snaps.length;
        render();
      }}, 500);
    }}

    DATA.samples.forEach((sample, idx) => {{
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = `sample ${{sample.sample_idx}}`;
      els.sampleSelect.appendChild(option);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-option";
      button.setAttribute("role", "option");
      button.dataset.sampleIndex = String(idx);

      const title = document.createElement("span");
      title.className = "sample-title";
      title.textContent = `sample ${{sample.sample_idx}}`;

      const detail = document.createElement("span");
      detail.className = "sample-detail";
      detail.textContent = sampleSummary(sample);

      const preview = document.createElement("span");
      preview.className = "sample-preview";
      preview.textContent = samplePreview(sample);

      button.append(title, detail, preview);
      button.addEventListener("click", () => selectSample(idx));
      els.sampleList.appendChild(button);
    }});
    els.sampleMeta.textContent = `${{DATA.sample_count}} sample${{DATA.sample_count === 1 ? "" : "s"}}`;
    els.snapshotMeta.textContent = `${{DATA.snapshot_count}} snapshots`;
    els.totalMeta.textContent = `${{DATA.total_steps}} steps`;
    els.sampleSelect.addEventListener("change", () => {{
      selectSample(Number(els.sampleSelect.value));
    }});
    els.slider.addEventListener("input", () => {{
      state.idx = Number(els.slider.value);
      render();
    }});
    els.prev.addEventListener("click", () => {{
      state.idx = Math.max(0, state.idx - 1);
      render();
    }});
    els.next.addEventListener("click", () => {{
      state.idx = Math.min(currentSnapshots().length - 1, state.idx + 1);
      render();
    }});
    els.play.addEventListener("click", () => setPlaying(!state.timer));
    window.addEventListener("resize", drawPlot);
    render();
  </script>
</body>
</html>
"""
    output = Path(path).expanduser()
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return str(output.resolve())


__all__ = [
    "config_get",
    "decode_ids_with_masks",
    "load_model_and_tokenizer",
    "load_yaml_config",
    "print_box",
    "print_decoded_batch_snapshot",
    "print_decoded_snapshot",
    "print_generation_result",
    "print_trace_snapshots",
    "report_weight_health",
    "resolve_dtype",
    "write_trace_html",
]

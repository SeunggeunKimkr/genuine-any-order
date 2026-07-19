"""Render a trajectory .pt record into a self-contained interactive HTML.

Delegates the actual HTML emission to ``flexmdm.utils.write_trace_html`` so
that the output matches the format consumed by ``evals/tree_analysis`` (a
``const DATA = {...}`` JSON blob with full per-step token state per sample).

The .pt record format (produced by ``evals.humaneval_compare.generate``) is:

  prompt:           str           — raw prompt text fed to the encoder
  prompt_len:       int           — number of prefix tokens (model input region)
  sequences:        (S, L) int32  — full token state at each of S snapshots
  attention_masks:  (S, L) bool   — active positions at each snapshot
  insertion_masks:  (S, L) bool   — freshly inserted positions (FlexMDM only;
                                    all-False for Dream-Coder, kept for
                                    compatibility but not surfaced in HTML)
  mask_id:          int
  pad_id:           int
  meta:             dict          — generation hyperparams, must include a
                                    ``tokenizer_source`` key (path or HF id)
                                    if no explicit tokenizer is passed.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from transformers import AutoTokenizer

from flexmdm.utils import write_trace_html


def _resolve_tokenizer(
    tokenizer_or_name: Any | None,
    *,
    record_meta: dict,
) -> Any:
    """Accept a real tokenizer object, a path/id string, or None (use meta).

    The ``write_trace_html`` payload uses ``tokenizer.decode``; any object
    with that method works.
    """
    if tokenizer_or_name is not None and not isinstance(tokenizer_or_name, str):
        return tokenizer_or_name
    name = (
        tokenizer_or_name
        or record_meta.get("tokenizer_source")
        or "Dream-org/Dream-Coder-v0-Base-7B"
    )
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def render_record_to_html(
    record_path: str,
    output_path: str,
    *,
    tokenizer_name: str | None = None,
    tokenizer: Any | None = None,
    mask_text: str = "[M]",
) -> str:
    """Load a .pt trajectory record and render it as interactive HTML.

    Parameters
    ----------
    record_path : str
        Path to a .pt produced by ``evals.humaneval_compare.generate``.
    output_path : str
        Path where the HTML viewer is written.
    tokenizer_name : str, optional
        Path / HF id used to load a tokenizer if ``tokenizer`` is not given.
    tokenizer : Any, optional
        An already-loaded tokenizer object. Takes precedence.
    mask_text : str
        Visible string used to render mask tokens. Default ``"[M]"`` matches
        what tree_analysis expects.

    Returns
    -------
    str
        Resolved path to the written HTML file.
    """
    rec = torch.load(record_path, map_location="cpu", weights_only=False)
    sequences = rec["sequences"].cpu().long()                       # (S, L)
    attention_masks = rec["attention_masks"].cpu().bool()           # (S, L)
    mask_id = int(rec["mask_id"])
    prompt_len = int(rec["prompt_len"])
    prompt_text = str(rec.get("prompt", ""))
    meta = dict(rec.get("meta", {}))

    if sequences.ndim != 2 or attention_masks.ndim != 2:
        raise ValueError(
            f"sequences and attention_masks must be (S, L); got "
            f"{tuple(sequences.shape)} and {tuple(attention_masks.shape)}."
        )

    s_count, length = sequences.shape
    if attention_masks.shape != sequences.shape:
        raise ValueError(
            f"attention_masks shape {tuple(attention_masks.shape)} != "
            f"sequences shape {tuple(sequences.shape)}."
        )
    if not (0 <= prompt_len <= length):
        raise ValueError(f"prompt_len {prompt_len} out of [0, {length}].")

    tok = _resolve_tokenizer(tokenizer or tokenizer_name, record_meta=meta)

    history = [sequences[s : s + 1] for s in range(s_count)]
    attention_history = [attention_masks[s : s + 1] for s in range(s_count)]
    history_steps = list(range(s_count))
    total_steps = int(meta.get("steps", max(0, s_count - 1)))

    prompt_mask = torch.zeros((1, length), dtype=torch.bool)
    prompt_mask[0, :prompt_len] = True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return write_trace_html(
        output_path,
        tokenizer=tok,
        history=history,
        attention_history=attention_history,
        history_steps=history_steps,
        total_steps=total_steps,
        prompts=[prompt_text],
        prompt_mask=prompt_mask,
        mask_id=mask_id,
        mask_text=mask_text,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, help="Path to a .pt record")
    parser.add_argument("--output", required=True, help="Path for the .html viewer")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/id; falls back to record meta["
        "'tokenizer_source'].",
    )
    parser.add_argument("--mask-text", default="[M]")
    args = parser.parse_args()
    out = render_record_to_html(
        args.record,
        args.output,
        tokenizer_name=args.tokenizer,
        mask_text=args.mask_text,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

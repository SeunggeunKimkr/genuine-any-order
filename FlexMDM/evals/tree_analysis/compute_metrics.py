"""Compute any-order metrics (CBC / RUB / OBW) directly from .pt traces.

We read the trace records produced by ``evals.humaneval_compare.generate``
(via ``record_path`` from ``common.py``). Each record is a dict with::

    prompt: str
    prompt_len: int
    sequences: int32 (S, L)         # full token state at every snapshot
    attention_masks: bool (S, L)    # active positions per snapshot
    insertion_masks: bool (S, L)    # positions just inserted at that step
    mask_id: int
    pad_id: int
    meta: dict

Algorithm:

1. Decode the body of the *final* sequence (``sequences[-1, prompt_len:final_active_len]``)
   with a tokenizer, strip trace markers, parse with
   :func:`evals.tree_analysis.visualize_final_ast_mapping.parse_with_fallbacks`.
2. Build the AST + visualization tree, tokenize the parsed source via the
   std-lib ``tokenize`` module, and align Python tokens to viz nodes.
3. For each *position* in the final sequence's body, find the step at which
   that position was *first* revealed (i.e., active and non-mask). For
   FlexMDM this requires reverse-walking the lineage through the
   ``insertion_masks`` so we know which step-s position corresponds to
   which final position. For DC-base ``insertion_masks`` is all False so
   the lineage is the identity.
4. Each .pt body position has a character span in the decoded source.
   Map Python tokens (whose char spans live in the *parsed* source) to
   .pt positions whose char spans (in the *decoded* source) overlap them.
   The Python token's reveal step is the **max** reveal step across all
   overlapping .pt positions (the token is "ready" only once every
   underlying piece is filled in).
5. Build a per-step ``generation_progress.entries`` list compatible with
   :func:`compute_tree_order_metrics_variant` and run the metric.

All the metric maths is delegated to the ported
``visualize_final_ast_mapping`` module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from evals.humaneval_compare.common import (
    Task,
    record_path,
)
from evals.tree_analysis.visualize_final_ast_mapping import (
    aggregate_tree_order_metrics,
    ast_to_record_tree,
    build_visualization_tree,
    clean_generated_answer,
    compute_tree_order_metrics_variant,
    countable_token_indices,
    line_start_offsets,
    line_texts,
    map_tokens_to_nodes,
    parse_with_fallbacks,
    spans_intersect,
    summarize_sample,
    tokenize_source,
)


# ---------------------------------------------------------------------------
# Lineage tracking: for each step s, which final-sequence position does
# each *active* position p of sequences[s] correspond to?
# ---------------------------------------------------------------------------


def _final_positions_per_step(
    attention_masks: torch.Tensor,    # (S, L) bool
    insertion_masks: torch.Tensor,    # (S, L) bool
) -> List[List[int]]:
    """For each step, return a list of length ``A_s`` (= active count at s)
    giving the final-step position each active position originated from.

    Output: ``out[s][i]`` = final-step position index corresponding to the
    i-th *active* position of step ``s``.

    Walks backwards from the last step. The convention follows
    ``flexmdm/inference.py:_apply_after_token_insertions``: at step s+1 the
    positions where ``insertion_masks[s+1]`` is True are *new* (no lineage
    in step s); the remaining active positions correspond — in *order* —
    to the active positions of step s.
    """
    if attention_masks.dtype != torch.bool:
        attention_masks = attention_masks.bool()
    if insertion_masks.dtype != torch.bool:
        insertion_masks = insertion_masks.bool()
    if attention_masks.shape != insertion_masks.shape:
        raise ValueError(
            f"attention_masks shape {tuple(attention_masks.shape)} != "
            f"insertion_masks shape {tuple(insertion_masks.shape)}"
        )
    n_steps, _length = attention_masks.shape

    # Lineage at each step: indices into sequences[-1].
    lineage_per_step: List[List[int]] = [[] for _ in range(n_steps)]

    # Final step: every active position maps to itself.
    final_active = attention_masks[-1].nonzero(as_tuple=False).flatten().tolist()
    lineage_per_step[-1] = list(final_active)

    # Walk backward.
    for s in range(n_steps - 1, 0, -1):
        # Active positions at step s, in order.
        active_s_positions = (
            attention_masks[s].nonzero(as_tuple=False).flatten().tolist()
        )
        # For each active position at step s, find whether it was inserted
        # at step s (in which case it has no parent in step s-1) or not
        # (in which case it carries over to step s-1).
        # The previous step's lineage = the subset of step-s lineage at
        # active positions where insertion_masks[s] is False, in order.
        prev_lineage: List[int] = []
        ins_s = insertion_masks[s]
        s_lineage = lineage_per_step[s]
        for active_idx, pos in enumerate(active_s_positions):
            if not bool(ins_s[pos].item()):
                prev_lineage.append(s_lineage[active_idx])
        lineage_per_step[s - 1] = prev_lineage

    return lineage_per_step


def _reveal_step_per_final_position(
    sequences: torch.Tensor,          # (S, L) int
    attention_masks: torch.Tensor,    # (S, L) bool
    insertion_masks: torch.Tensor,    # (S, L) bool
    *,
    mask_id: int,
    prompt_len: int,
) -> Dict[int, int]:
    """Map each final-sequence position p (in the answer body) to the
    earliest step at which that position is active and non-mask.

    Positions in [0, prompt_len) are skipped — they're prompt tokens, not
    part of the model's generation. The final step is guaranteed to be
    a fallback for any position never fully filled.
    """
    if attention_masks.dtype != torch.bool:
        attention_masks = attention_masks.bool()
    if insertion_masks.dtype != torch.bool:
        insertion_masks = insertion_masks.bool()
    n_steps, _length = sequences.shape
    final_active = (
        attention_masks[-1].nonzero(as_tuple=False).flatten().tolist()
    )
    body_final_positions = [p for p in final_active if p >= prompt_len]
    if not body_final_positions:
        return {}

    lineage_per_step = _final_positions_per_step(
        attention_masks, insertion_masks
    )

    reveal: Dict[int, Optional[int]] = {p: None for p in body_final_positions}
    for s in range(n_steps):
        active_s = attention_masks[s].nonzero(as_tuple=False).flatten().tolist()
        s_lineage = lineage_per_step[s]
        # Each active position p_s at step s corresponds to lineage position
        # s_lineage[idx] in the final sequence.
        for active_idx, p_s in enumerate(active_s):
            final_pos = s_lineage[active_idx]
            if final_pos < prompt_len:
                continue
            if reveal.get(final_pos) is not None:
                continue
            tok = int(sequences[s, p_s].item())
            if tok != mask_id:
                reveal[final_pos] = s
    # Fallback: any position never seen non-mask = revealed at the final step.
    for p in body_final_positions:
        if reveal[p] is None:
            reveal[p] = n_steps - 1
    return {p: int(s) for p, s in reveal.items() if s is not None}


# ---------------------------------------------------------------------------
# Final source decoding & per-position character span
# ---------------------------------------------------------------------------


@dataclass
class DecodedRecord:
    raw_decoded: str          # tokenizer.decode of the body (may include markers)
    cleaned_source: str       # markers stripped via clean_generated_answer
    body_positions: List[int] # final-sequence positions of body tokens
    position_char_spans: Dict[int, Tuple[int, int]]  # p -> (start, end) in raw_decoded
    raw_to_clean_offset: int  # raw_decoded.find(cleaned_source) (or -1)


def _decode_record_body(record: Dict[str, Any], tokenizer: Any) -> DecodedRecord:
    """Decode the answer body of the final sequence and return per-position
    character spans into the *raw* (markers-included) decoded string.

    ``cleaned_source`` is the model output with trace markers stripped via
    :func:`clean_generated_answer`. We also return the offset where
    ``cleaned_source`` is located inside ``raw_decoded`` so that callers
    can translate spans between the two coordinate systems.
    """
    seqs = record["sequences"]
    am = record["attention_masks"]
    pl = int(record["prompt_len"])
    final_seq = seqs[-1]
    final_am = am[-1].bool()
    real_len = int(final_am.sum().item())
    # Body positions = active positions of the final step >= prompt_len.
    active_final = final_am.nonzero(as_tuple=False).flatten().tolist()
    body_positions = [p for p in active_final if p >= pl][: real_len - pl]

    if not body_positions:
        return DecodedRecord(
            raw_decoded="",
            cleaned_source="",
            body_positions=[],
            position_char_spans={},
            raw_to_clean_offset=-1,
        )

    # Decode each token *individually* so we can locate per-position spans.
    # We then join them and use the running offsets as char spans. We use
    # ``skip_special_tokens=False`` here because we want to keep the
    # raw mask/marker text for diagnostic purposes; we'll strip it in
    # ``clean_generated_answer``.
    spans: Dict[int, Tuple[int, int]] = {}
    parts: List[str] = []
    cursor = 0
    for p in body_positions:
        tok_id = int(final_seq[p].item())
        text = tokenizer.decode([tok_id], skip_special_tokens=False)
        spans[p] = (cursor, cursor + len(text))
        parts.append(text)
        cursor += len(text)
    raw_decoded = "".join(parts)

    cleaned = clean_generated_answer(raw_decoded)
    raw_to_clean = raw_decoded.find(cleaned) if cleaned else -1
    return DecodedRecord(
        raw_decoded=raw_decoded,
        cleaned_source=cleaned,
        body_positions=body_positions,
        position_char_spans=spans,
        raw_to_clean_offset=raw_to_clean,
    )


# ---------------------------------------------------------------------------
# Token (Python tokenize) -> .pt position char span overlap
# ---------------------------------------------------------------------------


def _normalize_crlf(text: str) -> Tuple[str, List[int]]:
    """Replace ``\\r\\n`` with ``\\n`` and return ``(normalized, old_to_new)``
    where ``old_to_new[i]`` is the index in ``normalized`` corresponding to
    the boundary at position ``i`` in the original ``text``. Length is
    ``len(text) + 1`` so callers can translate exclusive ``end`` indices.
    """
    out: List[str] = []
    old_to_new: List[int] = []
    new_pos = 0
    i = 0
    while i < len(text):
        old_to_new.append(new_pos)
        if text[i] == '\r' and i + 1 < len(text) and text[i + 1] == '\n':
            # \r drops; \n moves into its slot. Both old indices i and i+1
            # map to the same new index for the resulting '\n'.
            old_to_new.append(new_pos)
            out.append('\n')
            new_pos += 1
            i += 2
        else:
            out.append(text[i])
            new_pos += 1
            i += 1
    old_to_new.append(new_pos)  # one-past-end
    return ''.join(out), old_to_new


def _python_token_reveal_steps(
    python_tokens: Sequence[Dict[str, Any]],
    decoded: DecodedRecord,
    position_reveal_steps: Dict[int, int],
    *,
    parse_source: str,
) -> Dict[int, Optional[int]]:
    """For each Python token (with char_start/char_end into ``parse_source``)
    return the step at which it's considered "revealed".

    A Python token is revealed at the **max** reveal-step over all
    final-sequence positions whose decoded char span overlaps the token's
    parse-source span. Both spans live in different coordinate systems
    (the decoded raw string vs. the cleaned/parse source). We translate
    via ``decoded.raw_to_clean_offset``: if the cleaned source is a
    contiguous substring of the raw decoded string at offset ``o``, then
    parse-source span ``[a, b]`` corresponds to raw decoded span
    ``[a + o, b + o]`` (provided ``parse_source == decoded.cleaned_source``).

    CRLF caveat: ``parse_with_fallbacks`` may use ``extract_longest_valid_code``
    which normalizes line endings (``splitlines() + '\\n'.join(...)``). When
    the original ``raw_decoded`` contains ``\\r\\n``, ``parse_source`` will
    have ``\\n`` and the substring lookup fails. We then fall back to
    matching against a CRLF-normalized ``raw_decoded`` and remap
    ``position_char_spans`` accordingly so token alignment still works.
    """
    reveal_by_token: Dict[int, Optional[int]] = {}

    raw_offset = decoded.raw_to_clean_offset
    parse_matches_clean = parse_source == decoded.cleaned_source
    position_char_spans = decoded.position_char_spans

    if not parse_matches_clean:
        idx = decoded.raw_decoded.find(parse_source)
        if idx < 0 and ('\r\n' in decoded.raw_decoded):
            # Line-ending normalization fallback.
            normalized_raw, old_to_new = _normalize_crlf(decoded.raw_decoded)
            idx = normalized_raw.find(parse_source)
            if idx >= 0:
                position_char_spans = {
                    p: (old_to_new[min(s, len(old_to_new) - 1)],
                        old_to_new[min(e, len(old_to_new) - 1)])
                    for p, (s, e) in decoded.position_char_spans.items()
                }
        raw_offset = idx if idx >= 0 else 0

    if raw_offset < 0:
        raw_offset = 0

    # Build a list of (final_pos, raw_start, raw_end, reveal_step).
    pos_spans: List[Tuple[int, int, int]] = []
    for final_pos, (raw_start, raw_end) in position_char_spans.items():
        step = position_reveal_steps.get(final_pos)
        if step is None:
            continue
        pos_spans.append((raw_start, raw_end, step))

    for token in python_tokens:
        token_index = int(token["token_index"])
        start = token.get("char_start")
        end = token.get("char_end")
        if start is None or end is None or end <= start:
            reveal_by_token[token_index] = None
            continue
        raw_token_start = int(start) + raw_offset
        raw_token_end = int(end) + raw_offset
        overlap_steps: List[int] = []
        for (rs, re_, step) in pos_spans:
            if spans_intersect(raw_token_start, raw_token_end, rs, re_):
                overlap_steps.append(step)
        reveal_by_token[token_index] = (
            max(overlap_steps) if overlap_steps else None
        )

    return reveal_by_token


# ---------------------------------------------------------------------------
# generation_progress.entries from per-token reveal steps
# ---------------------------------------------------------------------------


def build_generation_progress(
    tokens: Sequence[Dict[str, Any]],
    viz_nodes: Sequence[Dict[str, Any]],
    token_reveal_steps: Dict[int, Optional[int]],
    *,
    n_steps: int,
) -> Dict[str, Any]:
    """Translate per-Python-token reveal steps into the per-step entries
    list consumed by :func:`compute_tree_order_metrics_variant`.

    We emit one entry per *unique* reveal step that produced any countable
    Python token. Each entry's ``viz_completion`` records, per
    viz node id, the running ``generated`` / ``new`` / ``total`` counts.
    """
    countable = set(countable_token_indices(tokens))
    reveal_events: Dict[int, List[int]] = {}
    for token_index in countable:
        step = token_reveal_steps.get(int(token_index))
        if step is not None:
            reveal_events.setdefault(int(step), []).append(int(token_index))

    entry_steps = sorted(reveal_events)

    cumulative: set = set()
    entries: List[Dict[str, Any]] = []
    for progress_index, step in enumerate(entry_steps):
        new_tokens = set(reveal_events.get(step, []))
        cumulative |= new_tokens
        viz_completion: Dict[str, Dict[str, Any]] = {}
        for node in viz_nodes:
            node_tokens = set(
                node.get("subtree_token_indices") or node.get("token_indices") or []
            )
            node_tokens &= countable
            generated_count = len(node_tokens & cumulative)
            new_count = len(node_tokens & new_tokens)
            total_count = len(node_tokens)
            ratio = generated_count / total_count if total_count else 0.0
            viz_completion[str(node["viz_node_id"])] = {
                "generated": generated_count,
                "new": new_count,
                "total": total_count,
                "ratio": round(ratio, 4),
            }
        entries.append(
            {
                "progress_index": progress_index,
                "snapshot_index": step,
                "step": step,
                "progress": step / max(1, n_steps - 1),
                "visible_token_indices": sorted(cumulative),
                "generated_token_indices": sorted(cumulative),
                "new_token_indices": sorted(new_tokens),
                "viz_completion": viz_completion,
            }
        )

    return {
        "alignment_method": "pt-record final-position lineage; per-token reveal step "
        "= max reveal-step across overlapping final positions",
        "snapshot_count": n_steps,
        "unique_step_count": len(entries),
        "total_final_tokens": len(countable),
        "entries": entries,
    }


# ---------------------------------------------------------------------------
# Single-record analysis
# ---------------------------------------------------------------------------


METRIC_VARIANTS = ("reference_aware", "code_only", "entry_only")


def _empty_metrics_payload() -> Dict[str, Any]:
    empty_variant = {
        "variant": None,
        "local": [],
        "per_node": {},
        "aggregate": aggregate_tree_order_metrics([]),
        "excluded_reference_node_ids": [],
    }
    return {v: {**empty_variant, "variant": v} for v in METRIC_VARIANTS}


def _entry_point_subtree_viz_ids(
    viz_nodes: Sequence[Dict[str, Any]],
    *,
    tree,
    ast_obj_to_id: Dict[int, int],
    ast_to_viz: Dict[int, int],
    entry_point: str,
) -> Optional[set]:
    """Return the set of viz_node_ids reachable from the entry-point function.

    None if the entry-point function isn't defined in the tree (e.g. the
    model never emitted a ``def <entry_point>(...)``).
    """
    import ast as _ast
    target_viz_id: Optional[int] = None
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) \
                and node.name == entry_point:
            ast_id = ast_obj_to_id.get(id(node))
            if ast_id is not None:
                target_viz_id = ast_to_viz.get(ast_id)
            break
    if target_viz_id is None:
        return None
    nodes_by_id = {n["viz_node_id"]: n for n in viz_nodes}
    keep: set = set()
    stack = [target_viz_id]
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        node = nodes_by_id.get(nid)
        if node is None:
            continue
        keep.add(nid)
        for child in node.get("children", []) or []:
            stack.append(child)
    return keep


def analyze_record(
    record: Dict[str, Any],
    tokenizer: Any,
    *,
    sample_idx: Any = None,
    include_reference_nodes_variants: Sequence[bool] = (True, False),
) -> Dict[str, Any]:
    """Compute the full any-order analysis for one .pt record.

    Returns a dict with parse status, viz tree, per-token reveal steps,
    generation_progress, and tree_order_metrics for both
    ``reference_aware`` and ``code_only`` variants (or a subset, if
    ``include_reference_nodes_variants`` is restricted).
    """
    sequences = record["sequences"]
    attention_masks = record["attention_masks"]
    insertion_masks = record.get("insertion_masks")
    if insertion_masks is None:
        insertion_masks = torch.zeros_like(attention_masks, dtype=torch.bool)
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.long()
    else:
        sequences = torch.as_tensor(sequences, dtype=torch.long)
    if isinstance(attention_masks, torch.Tensor):
        attention_masks = attention_masks.bool()
    else:
        attention_masks = torch.as_tensor(attention_masks, dtype=torch.bool)
    if isinstance(insertion_masks, torch.Tensor):
        insertion_masks = insertion_masks.bool()
    else:
        insertion_masks = torch.as_tensor(insertion_masks, dtype=torch.bool)
    mask_id = int(record["mask_id"])
    prompt_len = int(record["prompt_len"])
    n_steps = int(sequences.shape[0])

    decoded = _decode_record_body(
        {
            "sequences": sequences,
            "attention_masks": attention_masks,
            "prompt_len": prompt_len,
        },
        tokenizer,
    )
    tree, parse_source, parse_error, parse_source_kind = parse_with_fallbacks(
        decoded.cleaned_source
    )
    starts = line_start_offsets(parse_source)
    lines = line_texts(parse_source)
    tokens, tokenize_error = tokenize_source(parse_source, starts, lines)

    base_result: Dict[str, Any] = {
        "sample_idx": sample_idx,
        "raw_final_answer": decoded.raw_decoded,
        "cleaned_final_answer": decoded.cleaned_source,
        "parsed_source": parse_source,
        "parse_source_kind": parse_source_kind,
        "parse_success": tree is not None,
        "parse_error": parse_error,
        "tokenize_error": tokenize_error,
        "tokens": tokens,
        "ast_nodes": [],
        "visualization_nodes": [],
        "token_to_deep_node": {},
        "token_to_visualization_node": {},
        "docstring_handling_info": {"docstring_collapsed": False, "records": []},
        "test_case_detection_info": {
            "has_generated_tests_or_examples": False,
            "nodes": [],
        },
        "trace_info": {
            "n_steps": n_steps,
            "prompt_len": prompt_len,
            "final_active_len": int(attention_masks[-1].sum().item()),
            "final_body_len": len(decoded.body_positions),
            "has_insertions": bool(insertion_masks.any().item()),
        },
        "generation_progress": {
            "alignment_method": "not_computed",
            "snapshot_count": n_steps,
            "unique_step_count": 0,
            "total_final_tokens": 0,
            "entries": [],
        },
        "python_token_reveal_steps": {},
        "tree_order_metrics": _empty_metrics_payload(),
    }

    if tree is None:
        return base_result

    ast_records, ast_obj_to_id, _, ast_parent = ast_to_record_tree(tree, parse_source)
    viz_nodes, ast_to_viz, doc_info, test_info = build_visualization_tree(
        tree, parse_source, ast_records, ast_obj_to_id
    )
    token_to_deep, token_to_viz = map_tokens_to_nodes(
        tokens, ast_records, viz_nodes, ast_parent, ast_to_viz
    )

    position_reveal_steps = _reveal_step_per_final_position(
        sequences,
        attention_masks,
        insertion_masks,
        mask_id=mask_id,
        prompt_len=prompt_len,
    )
    python_token_reveal = _python_token_reveal_steps(
        tokens,
        decoded,
        position_reveal_steps,
        parse_source=parse_source,
    )

    generation_progress = build_generation_progress(
        tokens,
        viz_nodes,
        python_token_reveal,
        n_steps=n_steps,
    )

    metrics: Dict[str, Any] = {}
    for include_reference in include_reference_nodes_variants:
        key = "reference_aware" if include_reference else "code_only"
        metrics[key] = compute_tree_order_metrics_variant(
            viz_nodes,
            generation_progress,
            include_reference_nodes=include_reference,
        )

    # entry_only variant: restrict to the AST subtree rooted at the
    # entry-point function. Excludes top-level imports, trailing print
    # statements, ``if __name__ == "__main__":`` blocks, etc. that the
    # model may have emitted around the actual answer function.
    entry_point = str(record.get("meta", {}).get("entry_point", "") or "")
    if entry_point:
        keep_ids = _entry_point_subtree_viz_ids(
            viz_nodes,
            tree=tree,
            ast_obj_to_id=ast_obj_to_id,
            ast_to_viz=ast_to_viz,
            entry_point=entry_point,
        )
        if keep_ids:
            entry_viz_nodes = [n for n in viz_nodes if n["viz_node_id"] in keep_ids]
            metrics["entry_only"] = compute_tree_order_metrics_variant(
                entry_viz_nodes,
                generation_progress,
                include_reference_nodes=False,
            )

    # Fill in defaults for any variant the caller didn't request /
    # couldn't produce (e.g., entry_only when the model didn't define
    # the entry-point function).
    for key in METRIC_VARIANTS:
        metrics.setdefault(
            key,
            {
                "variant": key,
                "local": [],
                "per_node": {},
                "aggregate": aggregate_tree_order_metrics([]),
                "excluded_reference_node_ids": [],
            },
        )

    base_result.update(
        {
            "tokens": tokens,
            "ast_nodes": ast_records,
            "visualization_nodes": viz_nodes,
            "token_to_deep_node": token_to_deep,
            "token_to_visualization_node": token_to_viz,
            "docstring_handling_info": doc_info,
            "test_case_detection_info": test_info,
            "generation_progress": generation_progress,
            "python_token_reveal_steps": {
                str(token_index): step
                for token_index, step in python_token_reveal.items()
            },
            "tree_order_metrics": metrics,
        }
    )
    return base_result


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------


def _result_for_disk(result: Dict[str, Any]) -> Dict[str, Any]:
    """Drop large per-token arrays from the result before writing to disk.

    Per-sample JSONs are intended to be readable scoreboards; the full
    AST node lists / token list explode disk usage and aren't useful
    downstream (everything we need for cross-sample analysis lives in
    ``tree_order_metrics`` and the summary stats from ``summarize_sample``).
    """
    keep = {
        "sample_idx",
        "parse_success",
        "parse_error",
        "parse_source_kind",
        "parsed_source",
        "tokenize_error",
        "trace_info",
        "tree_order_metrics",
        "generation_progress",
        "python_token_reveal_steps",
        "summary",
    }
    out: Dict[str, Any] = {k: result[k] for k in keep if k in result}
    return out


def per_sample_json_path(
    output_root: str,
    gen_dataset: str,
    task_id: str,
    model: str,
    alg: str,
    sample_k: int,
) -> str:
    safe_task = task_id.replace("/", "_")
    folder = os.path.join(output_root, "anyorder", "per_sample")
    fname = (
        f"{gen_dataset}__{safe_task}__{model}__{alg}__sample_{sample_k:02d}.json"
    )
    return os.path.join(folder, fname)


def summary_json_path(output_root: str) -> str:
    return os.path.join(output_root, "anyorder", "summary.json")


def analyze_sample_to_disk(
    *,
    output_root: str,
    gen_dataset: str,
    task: Task,
    model: str,
    alg: str,
    sample_k: int,
    tokenizer: Any,
    include_reference_nodes_variants: Sequence[bool] = (True, False),
) -> Optional[Dict[str, Any]]:
    """Run the analysis on one .pt record and write a per-sample JSON.

    Returns the trimmed on-disk dict (with a ``summary`` field), or
    ``None`` if the .pt is missing.
    """
    rp = record_path(output_root, gen_dataset, task.task_id, model, alg, sample_k)
    if not os.path.isfile(rp):
        return None
    record = torch.load(rp, map_location="cpu", weights_only=False)
    sample_uid = (
        f"{gen_dataset}__{task.task_id}__{model}__{alg}__sample_{sample_k:02d}"
    )
    result = analyze_record(
        record,
        tokenizer,
        sample_idx=sample_uid,
        include_reference_nodes_variants=include_reference_nodes_variants,
    )
    result["summary"] = summarize_sample(result)
    out_path = per_sample_json_path(
        output_root, gen_dataset, task.task_id, model, alg, sample_k
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    on_disk = _result_for_disk(result)
    on_disk["task_id"] = task.task_id
    on_disk["gen_dataset"] = gen_dataset
    on_disk["model"] = model
    on_disk["alg"] = alg
    on_disk["sample_k"] = sample_k
    with open(out_path, "w") as fh:
        json.dump(on_disk, fh, indent=2)
    return on_disk


# ---------------------------------------------------------------------------
# Cross-sample aggregation
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return float(sum(nums) / len(nums)) if nums else None


def aggregate_per_sample_jsons(
    per_sample_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Group per-sample results by (gen_dataset, model, alg) and produce
    aggregate CBC / RUB / OBW means over samples.

    Each key in the returned dict is ``f"{gen_dataset}|{model}|{alg}"``
    and maps to a dict with overall / split_only means, per variant.
    """
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in per_sample_results:
        key = (
            str(row.get("gen_dataset", "")),
            str(row.get("model", "")),
            str(row.get("alg", "")),
        )
        grouped.setdefault(key, []).append(row)

    out: Dict[str, Any] = {}
    for (gd, model, alg), rows in grouped.items():
        bucket: Dict[str, Any] = {
            "n_samples": len(rows),
            "n_parsed": sum(1 for r in rows if r.get("parse_success")),
        }
        for variant in METRIC_VARIANTS:
            agg = {}
            for level in ("overall", "split_only"):
                for metric in ("cbc", "rub", "rub_plus", "obw"):
                    vals = [
                        (r.get("tree_order_metrics", {})
                          .get(variant, {})
                          .get("aggregate", {})
                          .get(level, {})
                          .get(metric))
                        for r in rows
                    ]
                    agg[f"{level}_{metric}"] = _safe_mean(vals)
            bucket[variant] = agg
        out[f"{gd}|{model}|{alg}"] = bucket
    return out

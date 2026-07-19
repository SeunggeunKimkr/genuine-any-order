"""Inference utilities for FlexMDM."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn.functional as F

from .schedules import schedule_alpha, schedule_alpha_inverse, schedule_hazard_rate

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tiny fallback for minimal envs.
    def tqdm(iterable, **_kwargs):
        return iterable


POISSON_INSERTION_COUNT_SAMPLER = "poisson"
FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER = "floor_bernoulli"


def _sample_floor_plus_bernoulli(expected_count: torch.Tensor) -> torch.Tensor:
    """Sample floor(r) + Bernoulli(frac(r)) for non-negative expected counts."""
    expected_count = expected_count.float()
    base_count = torch.floor(expected_count)
    fractional_count = expected_count - base_count
    bernoulli_count = torch.bernoulli(fractional_count)
    return (base_count + bernoulli_count).long()


def _normalize_insertion_count_sampler(sampler: Optional[str]) -> str:
    if sampler is None:
        return POISSON_INSERTION_COUNT_SAMPLER
    normalized = str(sampler).strip().lower().replace("-", "_")
    if normalized == POISSON_INSERTION_COUNT_SAMPLER:
        return POISSON_INSERTION_COUNT_SAMPLER
    if normalized in {
        FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER,
        "bernoulli",
        "floor_plus_bernoulli",
    }:
        return FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER
    raise ValueError(
        "insertion_count_sampler must be one of "
        "{'poisson', 'floor_bernoulli'}, "
        f"got {sampler!r}."
    )


def _sample_insertion_counts(
    expected_count: torch.Tensor,
    *,
    insertion_count_sampler: str,
) -> torch.Tensor:
    """Draw integer insertion counts from a non-negative expected count tensor."""
    sampler = _normalize_insertion_count_sampler(insertion_count_sampler)
    if sampler == POISSON_INSERTION_COUNT_SAMPLER:
        return torch.poisson(expected_count).long()
    if sampler == FLOOR_BERNOULLI_INSERTION_COUNT_SAMPLER:
        return _sample_floor_plus_bernoulli(expected_count)
    raise AssertionError(f"Unhandled insertion count sampler {sampler!r}")


def gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample token ids with Gumbel-max; temperature=0 gives argmax."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits.float() / float(temperature)
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise.clamp_min(1e-8)) + 1e-8)
    return (logits + gumbel).argmax(dim=-1)


def _sample_x0_and_confidence(
    logits: torch.Tensor,
    *,
    temperature: float,
    confidence_method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample x0 first, then score each sampled clean token's confidence.

    Ranking methods supported (used by `_adaptive_unmask_step` to decide which
    masked positions to reveal this step):
    - ``top_k`` / ``top_k_probability``: confidence = P(sampled_token) under
      softmax(logits). Matches Dream-Coder's ``maskgit_plus`` alg.
    - ``entropy``: confidence = -H(softmax(logits)). Matches Dream-Coder's
      ``entropy`` alg: low-entropy (i.e. confident) positions reveal first.

    Returns ``(sampled_tokens, confidence, probs)``; callers that need the full
    per-position softmax (e.g. the long-segment force-unmask trick) use
    ``probs`` directly instead of recomputing it.
    """
    sampled_tokens = gumbel_sample(logits, temperature=temperature)
    probs = F.softmax(logits.float(), dim=-1)

    if confidence_method in {"top_k", "top_k_probability"}:
        sampled_prob = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        confidence = sampled_prob
    elif confidence_method == "entropy":
        log_probs = probs.clamp_min(1e-30).log()
        neg_entropy = (probs * log_probs).sum(dim=-1)  # == -H, higher = more confident
        confidence = neg_entropy
    else:
        raise ValueError(
            "confidence_method must be one of "
            "{'top_k', 'top_k_probability', 'entropy', 'origin'}, "
            f"got {confidence_method!r}."
        )

    return sampled_tokens, confidence, probs


def _find_force_unmask_choices(
    probs_row: torch.Tensor,
    mask_row: torch.Tensor,
    selected_row: torch.Tensor,
    *,
    min_segment_len: int,
    top_k_per_position: int,
) -> list[tuple[int, int]]:
    """Positional-uncertainty unstick trick (long-segment force-unmask).

    For each maximal run of currently masked positions ``m_1..m_L`` (1-based)
    with ``L >= min_segment_len`` where the ordinary confidence step picked no
    position in ``m_2..m_{L-1}``, slide a length-4 window over ``m_3..m_{L-2}``
    and, within each window:
      1. take the union of top-``K`` token ids across the 4 positions,
      2. for each candidate ``v``, sum ``p_j(v)`` across the 4 window positions,
      3. pick the token ``v*`` with the largest total mass.
    Across the windows in one segment, pick the ``(window, v*)`` pair with the
    highest total mass, then commit at ``j* = argmax_{j in window} p_j(v*)``.

    Returns one ``(absolute_position, token_id)`` tuple per qualifying segment.
    """
    SUBSEG = 4
    out: list[tuple[int, int]] = []

    mask_list = mask_row.tolist()
    L_buf = len(mask_list)

    i = 0
    while i < L_buf:
        if not mask_list[i]:
            i += 1
            continue
        start = i
        while i < L_buf and mask_list[i]:
            i += 1
        end = i  # exclusive
        seg_len = end - start
        if seg_len < min_segment_len:
            continue
        # Interior check: m_2..m_{L-1} maps to abs slice [start+1, end-1).
        if selected_row[start + 1:end - 1].any().item():
            continue
        # Subsegment offset s within the segment: 1-based i in [3, L-5] gives
        # 0-based s in [2, seg_len-6]. Window positions are [start+s, start+s+3].
        s_max = seg_len - 6
        if s_max < 2:
            continue

        seg_best: Optional[tuple[float, int, int]] = None  # (mass, abs_j, token)
        for s in range(2, s_max + 1):
            abs_lo = start + s
            sub_probs = probs_row[abs_lo:abs_lo + SUBSEG]  # (4, V)
            cand = (
                sub_probs.topk(k=top_k_per_position, dim=-1)
                .indices.flatten()
                .unique()
            )
            cand_probs = sub_probs.index_select(dim=-1, index=cand)  # (4, |C|)
            totals = cand_probs.sum(dim=0)  # (|C|,)
            mass_t, idx_t = totals.max(dim=0)
            v = int(cand[idx_t].item())
            j_local = int(sub_probs[:, v].argmax().item())
            mass = float(mass_t.item())
            abs_j = abs_lo + j_local
            if seg_best is None or mass > seg_best[0]:
                seg_best = (mass, abs_j, v)

        if seg_best is not None:
            out.append((seg_best[1], seg_best[2]))

    return out


# Characters that — on their own — make a token NOT meaningful.
# Whitespace + structural punctuation. Operator chars (+ - * / % = < > & | ^ ~ !)
# are intentionally excluded (i.e. tokens containing them are meaningful).
_NON_MEANINGFUL_CHARS = frozenset(
    " \t\n\r\v\f  ​()[]{}.,:;'\"`@#$?\\"
)


def build_meaningful_token_mask(
    tokenizer: Any,
    vocab_size: int,
    *,
    extra_excluded_ids: Optional[list[int]] = None,
    extra_excluded_strings: Optional[list[str]] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Boolean mask over the tokenizer vocabulary marking *meaningful* tokens.

    A token id is "meaningful" iff its decoded string is non-empty and contains
    at least one character outside ``_NON_MEANINGFUL_CHARS``. So whitespace-only
    tokens (single space, multi-space, tabs, newlines), pure structural
    punctuation (``,``, ``.``, ``(``, ``)``, ``[``, ``]``, ``{``, ``}``, ``:``,
    ``;``, quotes, etc.) are excluded; identifiers, keywords, numbers, and
    operator tokens (``+``, ``-=``, ``!=``, ``**``...) are kept.

    Special tokens (``tokenizer.all_special_ids``) and any ``extra_excluded_ids``
    are forced to False. ``extra_excluded_strings`` is a list of strings: any
    token whose ``tokenizer.decode([tid]).strip()`` matches one of these
    strings (case-sensitive) is also forced to False — this lets callers
    blacklist e.g. ``["print"]`` to stop the meaningful-replace trick from
    anchoring on a stub builtin that's already in the prompt.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    bad_strings = set(extra_excluded_strings or [])
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s:
            continue
        if all((ch in _NON_MEANINGFUL_CHARS) for ch in s):
            continue
        if bad_strings and s.strip() in bad_strings:
            continue
        mask[tid] = True
    special_ids = list(getattr(tokenizer, "all_special_ids", []) or [])
    if extra_excluded_ids:
        special_ids.extend(int(i) for i in extra_excluded_ids)
    for sid in special_ids:
        if 0 <= int(sid) < vocab_size:
            mask[int(sid)] = False
    if device is not None:
        mask = mask.to(device)
    return mask


def _find_meaningful_replace_candidate(
    probs_row: torch.Tensor,           # (L, V) raw softmax probs (pre-temperature)
    mask_row: torch.Tensor,            # (L,) bool — currently masked positions
    selected_row: torch.Tensor,        # (L,) bool — positions ordinary unmask picked
    meaningful_mask: torch.Tensor,     # (V,) bool — True for meaningful token ids
    *,
    min_segment_len: int = 20,
    window_len: int = 5,
) -> Optional[tuple[int, int, float]]:
    """Find the best (position, token_id, mass) candidate for the meaningful-
    replace trick across all maximal mask runs of length >= ``min_segment_len``.

    For each run with length ``L`` at positions ``[start, end)``:
    1. Define non-boundary region ``[start+2, end-3]`` inclusive (excluding 2
       positions on each side).
    2. If the ordinary unmask already chose any position in the non-boundary
       region, skip this run.
    3. Slide a window of length ``window_len`` over the non-boundary region. In
       each window, take the meaningful token with the largest sum of probs
       across the window (raw probs, no temperature). Across windows in this
       run, keep the (window, token, mass) with the largest summed mass.
    4. Across all qualifying runs, return the global best ``(absolute_position,
       token_id, summed_mass)`` or ``None`` if no run qualifies.

    Position within the winning window is the per-position argmax of that
    token's prob (i.e. commit the meaningful token at the position where it has
    the highest single-position likelihood within its winning window).
    """
    L = probs_row.shape[0]
    V = probs_row.shape[1]
    if meaningful_mask.shape[0] != V:
        raise ValueError(
            f"meaningful_mask has size {meaningful_mask.shape[0]}, "
            f"expected vocab_size={V}"
        )
    mask_list = mask_row.tolist()
    best: Optional[tuple[float, int, int]] = None  # (mass, abs_pos, token_id)

    i = 0
    while i < L:
        if not mask_list[i]:
            i += 1
            continue
        start = i
        while i < L and mask_list[i]:
            i += 1
        end = i  # exclusive
        seg_len = end - start
        if seg_len < min_segment_len:
            continue
        # Non-boundary region: [start+2, end-3] inclusive == slice [start+2 : end-2]
        nb_start = start + 2
        nb_end = end - 2
        nb_len = nb_end - nb_start
        if nb_len < window_len:
            continue
        # Skip if ordinary unmask chose any position inside the non-boundary region
        if bool(selected_row[nb_start:nb_end].any().item()):
            continue
        nb_probs = probs_row[nb_start:nb_end].float()  # (nb_len, V) — float for sum stability
        # Sliding-`window_len` sum via cumsum trick.
        zero_pad = torch.zeros_like(nb_probs[:1])
        cumsum = torch.cat([zero_pad, nb_probs], dim=0).cumsum(dim=0)  # (nb_len+1, V)
        window_mass = cumsum[window_len:] - cumsum[:-window_len]       # (nb_len-window_len+1, V)
        # Restrict to meaningful tokens; non-meaningful set to -inf.
        neg_inf = torch.full((), float("-inf"), dtype=window_mass.dtype, device=window_mass.device)
        window_mass = torch.where(
            meaningful_mask.unsqueeze(0).to(window_mass.device),
            window_mass,
            neg_inf,
        )
        per_window_mass, per_window_token = window_mass.max(dim=1)  # (#windows,) each
        best_win_mass, best_win_idx = per_window_mass.max(dim=0)
        if not torch.isfinite(best_win_mass):
            continue
        mass_val = float(best_win_mass.item())
        token_v = int(per_window_token[best_win_idx].item())
        win_idx = int(best_win_idx.item())
        # Commit position: argmax of p_j(v) within the winning window.
        win_probs_for_v = nb_probs[win_idx:win_idx + window_len, token_v]  # (window_len,)
        pos_in_window = int(win_probs_for_v.argmax().item())
        abs_pos = nb_start + win_idx + pos_in_window
        if best is None or mass_val > best[0]:
            best = (mass_val, abs_pos, token_v)

    if best is None:
        return None
    mass, pos, token = best
    return (pos, token, mass)


def _adaptive_unmask_step(
    xt: torch.Tensor,
    logits: torch.Tensor,
    mask_indices: torch.Tensor,
    *,
    t: torch.Tensor,
    dt: float,
    temperature: float,
    confidence_method: str,
    eps: float,
    hazard_rate_schedule: str,
    hazard_rate_exponent: Optional[float] = None,
    hazard_rate_params: Optional[Mapping[str, Any]] = None,
    force_unmask_long_segments: bool = False,
    force_unmask_min_segment_len: int = 9,
    force_unmask_top_k_per_position: int = 10,
    meaningful_replace: bool = False,
    meaningful_replace_min_segment_len: int = 20,
    meaningful_replace_window_len: int = 5,
    meaningful_replace_min_step: int = 100,
    meaningful_token_mask: Optional[torch.Tensor] = None,
    step_idx: int = 0,
    use_midpoint_hazard_rate: bool = False,
) -> torch.Tensor:
    """Confidence-based any-order unmasking.
    (1) first sample a clean token (2) then rank masked positions by the model

    When ``force_unmask_long_segments`` is True, additionally applies the
    long-segment positional-uncertainty trick (see
    ``_find_force_unmask_choices``): if the ordinary confidence step reveals no
    interior position of a mask run of length >= ``force_unmask_min_segment_len``,
    force one interior reveal chosen by the cross-position mass argmax over
    sliding length-4 windows.

    When ``meaningful_replace`` is True (and ``meaningful_token_mask`` is
    supplied), at each step *substitute* the lowest-prob ordinary commit with a
    meaningful-token candidate drawn from a long mask run, when that
    candidate's summed window mass exceeds the lowest-prob commit's
    single-position prob. See ``_find_meaningful_replace_candidate`` for the
    selection rule.
    """
    sampled_tokens, confidence, probs = _sample_x0_and_confidence(
        logits,
        temperature=temperature,
        confidence_method=confidence_method,
    )
    # just score mask tokens
    min_score = torch.finfo(confidence.dtype).min
    confidence = confidence.masked_fill(~mask_indices, min_score)

    # determine the number of tokens to unmask
    hazard_t = t + 0.5 * float(dt) if use_midpoint_hazard_rate else t
    rate = schedule_hazard_rate(
        hazard_t,
        schedule=hazard_rate_schedule,
        eps=eps,
        field="unmasking hazard_rate",
        exponent=hazard_rate_exponent,
        params=hazard_rate_params,
    )
    masked_count = mask_indices.sum(dim=1).to(dtype=rate.dtype)
    unmask_count = torch.poisson(masked_count * rate * float(dt)).long()

    new_xt = xt.clone()
    for batch_idx in range(xt.shape[0]):
        candidates = int(mask_indices[batch_idx].sum().item())
        k = min(int(unmask_count[batch_idx].item()), candidates)
        if k > 0:
            select_idx = confidence[batch_idx].topk(k=k).indices
            new_xt[batch_idx, select_idx] = sampled_tokens[batch_idx, select_idx]
        else:
            select_idx = None

        if force_unmask_long_segments:
            selected_row = torch.zeros_like(mask_indices[batch_idx])
            if select_idx is not None:
                selected_row[select_idx] = True
            forced = _find_force_unmask_choices(
                probs[batch_idx],
                mask_indices[batch_idx],
                selected_row,
                min_segment_len=force_unmask_min_segment_len,
                top_k_per_position=force_unmask_top_k_per_position,
            )
            for pos, token in forced:
                new_xt[batch_idx, pos] = token

        if (
            meaningful_replace
            and step_idx >= meaningful_replace_min_step
            and meaningful_token_mask is not None
            and select_idx is not None
            and select_idx.numel() > 0
        ):
            selected_row = torch.zeros_like(mask_indices[batch_idx])
            selected_row[select_idx] = True
            cand = _find_meaningful_replace_candidate(
                probs[batch_idx],
                mask_indices[batch_idx],
                selected_row,
                meaningful_token_mask,
                min_segment_len=meaningful_replace_min_segment_len,
                window_len=meaningful_replace_window_len,
            )
            if cand is not None:
                cand_pos, cand_tok, cand_mass = cand
                committed_tokens = sampled_tokens[batch_idx, select_idx]
                committed_probs = probs[batch_idx, select_idx, committed_tokens].float()
                v1_local = int(committed_probs.argmin().item())
                v1_pos = int(select_idx[v1_local].item())
                v1_prob = float(committed_probs[v1_local].item())
                if cand_mass > v1_prob:
                    # Restore the masked state at v_1's chosen position …
                    new_xt[batch_idx, v1_pos] = xt[batch_idx, v1_pos]
                    # … and commit the meaningful candidate at its winning slot.
                    new_xt[batch_idx, cand_pos] = cand_tok
    return new_xt


def _origin_unmask_step(
    xt: torch.Tensor,
    logits: torch.Tensor,
    mask_indices: torch.Tensor,
    *,
    t: torch.Tensor,
    dt: float,
    temperature: float,
    eps: float,
    hazard_rate_schedule: str,
    hazard_rate_exponent: Optional[float] = None,
    hazard_rate_params: Optional[Mapping[str, Any]] = None,
    use_midpoint_hazard_rate: bool = False,
) -> torch.Tensor:
    """Independent-per-position Bernoulli unmask (Dream-Coder 'origin' alg).

    Each masked position is independently replaced with a token sampled from
    softmax(logits/T) with probability ``p = 1 - exp(-rate(t) * dt)``. There is
    no cross-position ranking — the update is a plain continuous-time CTMC step
    driven by the schedule hazard rate, which is the continuous-time analogue
    of Dream's ``1 - s/t`` formula.
    """
    sampled_tokens = gumbel_sample(logits, temperature=temperature)
    hazard_t = t + 0.5 * float(dt) if use_midpoint_hazard_rate else t
    rate = schedule_hazard_rate(
        hazard_t,
        schedule=hazard_rate_schedule,
        eps=eps,
        field="unmasking hazard_rate",
        exponent=hazard_rate_exponent,
        params=hazard_rate_params,
    )
    p_transfer = 1.0 - torch.exp(-rate * float(dt))
    p_transfer = p_transfer.clamp(min=0.0, max=1.0).to(xt.device)
    p_broadcast = p_transfer.view(-1, 1).expand_as(mask_indices).float()
    fire = torch.rand_like(p_broadcast) < p_broadcast
    replace = mask_indices & fire
    new_xt = xt.clone()
    new_xt[replace] = sampled_tokens[replace]
    return new_xt


def _valid_insertion_gaps(
    *,
    prompt_len: torch.Tensor,
    xt_len: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return valid after-token insertion gaps.

    The B x L insertion head uses position i to mean "insert after token i".
    Prompt-internal gaps and padding positions are invalid. 
    """
    pos = torch.arange(max_len, device=device).view(1, max_len)
    first_allowed = (prompt_len - 1).clamp_min(0).view(-1, 1)
    real_token = pos < xt_len.view(-1, 1)
    after_prompt_boundary = pos >= first_allowed
    has_prompt_anchor = prompt_len.view(-1, 1) > 0
    return real_token & after_prompt_boundary & has_prompt_anchor


def _cap_insertions_to_capacity(
    insertions: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    """Greedily cap per-gap insertions so sequence length never exceeds L."""
    capped = torch.zeros_like(insertions)
    for batch_idx in range(insertions.shape[0]):
        remaining = int(capacity[batch_idx].item())
        if remaining <= 0:
            continue
        for pos_idx in range(insertions.shape[1]):
            count = min(int(insertions[batch_idx, pos_idx].item()), remaining)
            if count > 0:
                capped[batch_idx, pos_idx] = count
                remaining -= count
            if remaining <= 0:
                break
    return capped


def _apply_after_token_insertions(
    xt: torch.Tensor,
    attention_mask: torch.Tensor,
    insertions: torch.Tensor,
    *,
    pad_id: int,
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert mask tokens after each original token according to insertions.

    Also returns ``is_inserted`` (B, L) bool: True at positions in the new
    sequence that are freshly-inserted masks (as opposed to shifted-over
    originals). Consumers can use this to distinguish insertions from
    reveals when diffing consecutive snapshots.
    """
    batch_size, max_len = xt.shape
    device = xt.device
    pos = torch.arange(max_len, device=device).view(1, max_len).expand_as(xt)
    batch = torch.arange(batch_size, device=device).view(batch_size, 1).expand_as(xt)

    attention_mask = attention_mask.bool()
    xt_len = attention_mask.sum(dim=1)
    total_insertions = insertions.sum(dim=1)
    new_len = xt_len + total_insertions

    new_xt = torch.full_like(xt, pad_id)
    new_attention_mask = torch.zeros_like(attention_mask)
    nonpad_region = pos < new_len.view(batch_size, 1)
    new_xt[nonpad_region] = mask_id
    new_attention_mask[nonpad_region] = True

    # Start: every new-active position is assumed inserted; then flip back
    # the positions that receive a shifted original token.
    is_inserted = nonpad_region.clone()

    shift_before_token = insertions.cumsum(dim=1) - insertions
    new_pos = pos + shift_before_token
    original_token = pos < xt_len.view(batch_size, 1)
    new_xt[batch[original_token], new_pos[original_token]] = xt[original_token]
    is_inserted[batch[original_token], new_pos[original_token]] = False
    return new_xt, new_attention_mask, is_inserted


def _insertion_step(
    xt: torch.Tensor,
    attention_mask: torch.Tensor,
    log_length_pred: torch.Tensor,
    *,
    prompt_len: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    pad_id: int,
    mask_id: int,
    eps: float,
    hazard_rate_schedule: str,
    hazard_rate_exponent: Optional[float] = None,
    hazard_rate_params: Optional[Mapping[str, Any]] = None,
    insertion_count_sampler: str = POISSON_INSERTION_COUNT_SAMPLER,
    use_midpoint_hazard_rate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_len = xt.shape
    attention_mask = attention_mask.bool()
    xt_len = attention_mask.sum(dim=1)
    hazard_t = t + 0.5 * float(dt) if use_midpoint_hazard_rate else t
    rate = schedule_hazard_rate(
        hazard_t,
        schedule=hazard_rate_schedule,
        eps=eps,
        field="insertion hazard_rate",
        exponent=hazard_rate_exponent,
        params=hazard_rate_params,
    )
    insertion_rate = log_length_pred.clamp(max=15.0).float().exp() * rate[:, None]
    insertions = _sample_insertion_counts(
        insertion_rate * float(dt),
        insertion_count_sampler=insertion_count_sampler,
    )

    valid_gaps = _valid_insertion_gaps(
        prompt_len=prompt_len,
        xt_len=xt_len,
        max_len=max_len,
        device=xt.device,
    )
    insertions = insertions.masked_fill(~valid_gaps, 0)
    insertions = _cap_insertions_to_capacity(
        insertions,
        capacity=max_len - xt_len,
    )

    if insertions.sum().item() == 0:
        is_inserted = torch.zeros_like(attention_mask, dtype=torch.bool)
        return xt, attention_mask, is_inserted
    return _apply_after_token_insertions(
        xt,
        attention_mask,
        insertions,
        pad_id=pad_id,
        mask_id=mask_id,
    )


def generate(*args: Any, **kwargs: Any):
    """Generate with FlexMDM."""
    return flexmdm_generate(*args, **kwargs)


def _infer_prompt_attention_mask(
    input_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Infer prompt + internal PAD separator mask for prompt-only inference."""
    batch_size, max_len = input_ids.shape
    device = input_ids.device
    pos = torch.arange(max_len, device=device).view(1, max_len)
    prompt_token_len = input_ids.ne(pad_id).sum(dim=1)
    initial_len = torch.minimum(
        prompt_token_len + prompt_token_len.lt(max_len).long(),
        prompt_token_len.new_full(prompt_token_len.shape, max_len),
    )
    return pos < initial_len.view(batch_size, 1)


def _map_inference_time_to_model_time(
    event_t: torch.Tensor,
    *,
    train_insertion_schedule: str,
    train_insertion_exponent: Optional[float],
    train_insertion_params: Optional[Mapping[str, Any]],
    inference_insertion_schedule: str,
    inference_insertion_exponent: Optional[float],
    inference_insertion_params: Optional[Mapping[str, Any]],
) -> torch.Tensor:
    """Inference-time schedule reparameterization.

    Returns the training-time t-coordinate at which alpha equals the inference
    schedule's alpha at ``event_t``. The model is fed this `model_t` so it sees
    inputs at training-distribution time-coordinates even when the inference
    schedule differs from the training schedule.

    Concretely: gamma = inference_alpha(event_t); return train_alpha_inverse(gamma).
    """
    gamma = schedule_alpha(
        event_t,
        schedule=inference_insertion_schedule,
        field="inference insertion alpha",
        exponent=inference_insertion_exponent,
        params=inference_insertion_params,
    )
    return schedule_alpha_inverse(
        gamma,
        schedule=train_insertion_schedule,
        field="train insertion alpha inverse",
        exponent=train_insertion_exponent,
        params=train_insertion_params,
    )


@torch.no_grad()
def flexmdm_generate(
    model: torch.nn.Module,
    steps: int,
    input_ids: torch.Tensor,
    mask_id: int,
    pad_id: int,
    attention_mask: Optional[torch.Tensor] = None,
    prompt_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    confidence_method: str = "top_k",
    trace: bool = False,
    tokenizer: Optional[Any] = None,
    eps: float = 1e-6,
    insertion_schedule: str = "linear",
    unmasking_schedule: str = "linear",
    insertion_exponent: Optional[float] = None,
    unmasking_exponent: Optional[float] = None,
    insertion_params: Optional[Mapping[str, Any]] = None,
    unmasking_params: Optional[Mapping[str, Any]] = None,
    train_insertion_schedule: Optional[str] = None,
    train_insertion_exponent: Optional[float] = None,
    train_insertion_params: Optional[Mapping[str, Any]] = None,
    trace_callback: Optional[
        Callable[[int, int, torch.Tensor, torch.Tensor], None]
    ] = None,
    trace_every: int = 0,
    force_unmask_long_segments: bool = False,
    force_unmask_min_segment_len: int = 9,
    force_unmask_top_k_per_position: int = 10,
    meaningful_replace: bool = False,
    meaningful_replace_min_segment_len: int = 20,
    meaningful_replace_window_len: int = 5,
    meaningful_replace_min_step: int = 100,
    meaningful_token_mask: Optional[torch.Tensor] = None,
    insertion_count_sampler: str = POISSON_INSERTION_COUNT_SAMPLER,
    use_midpoint_hazard_rate: bool = False,
):
    """Batched FlexMDM sampler initialized from fixed prompt tokens.

    ``tokenizer`` is accepted for API compatibility and unused.

    ``force_unmask_long_segments`` enables the positional-uncertainty unstick
    trick (see ``_find_force_unmask_choices``) in confidence-based unmask steps
    (top_k / entropy). It is silently ignored for ``confidence_method='origin'``
    because the origin step is per-position Bernoulli, not ranking-based.
    The released evaluation runs do NOT use it (all trick flags default off).

    ``insertion_count_sampler`` selects how integer insertion counts are drawn
    from the expected per-gap rate ``insertion_rate * dt``: ``"poisson"`` (the
    historical default) or ``"floor_bernoulli"`` (deterministic floor plus a
    Bernoulli for the fractional remainder).
    """
    del tokenizer

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if trace_every < 0:
        raise ValueError(f"trace_every must be >= 0, got {trace_every}.")
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}."
        )
    if input_ids.eq(mask_id).any():
        raise ValueError("input_ids should not contain mask tokens initially.")

    # Resolve schedule reparameterization defaults: when train_* are not given,
    # fall back to the inference schedule (i.e. no reparameterization, model_t == event_t).
    if train_insertion_schedule is None:
        train_insertion_schedule = insertion_schedule
    if (
        train_insertion_exponent is None
        and train_insertion_schedule == insertion_schedule
    ):
        train_insertion_exponent = insertion_exponent
    if (
        train_insertion_params is None
        and train_insertion_schedule == insertion_schedule
    ):
        train_insertion_params = insertion_params

    device = input_ids.device
    xt = input_ids.clone().to(device)
    batch_size = xt.shape[0]
    dt = 1.0 / int(steps)
    t = torch.zeros(batch_size, device=device, dtype=torch.float32)
    if attention_mask is None:
        xt_attention_mask = _infer_prompt_attention_mask(xt, pad_id=pad_id)
    else:
        if attention_mask.shape != xt.shape:
            raise ValueError(
                "attention_mask shape "
                f"{tuple(attention_mask.shape)} must match input_ids shape {tuple(xt.shape)}"
            )
        xt_attention_mask = attention_mask.to(device=device).bool()
    if prompt_mask is None:
        prompt_mask = xt_attention_mask.clone()
    else:
        if prompt_mask.shape != xt.shape:
            raise ValueError(
                f"prompt_mask shape {tuple(prompt_mask.shape)} must match "
                f"input_ids shape {tuple(xt.shape)}"
            )
        prompt_mask = prompt_mask.to(device=device).bool()
    if (prompt_mask & ~xt_attention_mask).any():
        raise ValueError("prompt_mask must be a subset of attention_mask")

    prompt_len = prompt_mask.sum(dim=1)
    history = [] if trace else None
    attention_history = [] if trace else None
    # Per-step insertion mask aligned with `history`: entry s is (B, L) bool
    # marking positions that were freshly-inserted by the insertion step at
    # step s (step 0 is the pre-loop snapshot, which has no insertions).
    insertion_history = [] if trace else None

    # Scratch var so record_trace can publish the insertion mask applied this
    # step. Each iteration updates this right after the insertion_step call.
    last_is_inserted = torch.zeros_like(xt_attention_mask, dtype=torch.bool)

    def record_trace(completed_steps: int) -> None:
        if not trace:
            return
        assert history is not None
        assert attention_history is not None
        assert insertion_history is not None
        seq_snapshot = xt.detach().clone()
        attention_snapshot = xt_attention_mask.detach().clone()
        insertion_snapshot = last_is_inserted.detach().clone()
        history.append(seq_snapshot)
        attention_history.append(attention_snapshot)
        insertion_history.append(insertion_snapshot)
        if trace_callback is None:
            return
        should_emit = (
            trace_every <= 0
            or completed_steps == 0
            or completed_steps == steps
            or completed_steps % trace_every == 0
        )
        if should_emit:
            trace_callback(
                completed_steps,
                steps,
                seq_snapshot,
                attention_snapshot,
            )

    record_trace(0)

    for step_idx in tqdm(range(steps), desc="FlexMDM sampling"):
        # Schedule reparameterization: feed the model the training-time
        # coordinate where alpha matches the inference schedule's alpha at t.
        # When train_* == inference_*, model_t == t (no reparameterization).
        model_t = _map_inference_time_to_model_time(
            t,
            train_insertion_schedule=train_insertion_schedule,
            train_insertion_exponent=train_insertion_exponent,
            train_insertion_params=train_insertion_params,
            inference_insertion_schedule=insertion_schedule,
            inference_insertion_exponent=insertion_exponent,
            inference_insertion_params=insertion_params,
        )
        out = model(xt, model_t, attention_mask=xt_attention_mask)
        if isinstance(out, dict):
            logits = out["logits"]
            log_length_pred = out.get("log_length")
        else:
            logits = out
            log_length_pred = None
        if log_length_pred is None:
            raise ValueError("FlexMDM inference requires log-length predictions.")

        # Dream-style shift: position i predicts token i + 1.
        unmask_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_indices = xt.eq(mask_id) & xt_attention_mask

        if step_idx == steps - 1:
            final_tokens = unmask_logits.argmax(dim=-1)
            xt = xt.clone()
            xt[mask_indices] = final_tokens[mask_indices]
            if trace:
                # Final step has no insertion phase; the published mask is all
                # False by construction (last_is_inserted retains its previous
                # value which is irrelevant here; zero it explicitly).
                last_is_inserted = torch.zeros_like(  # noqa: F841 — used via closure
                    xt_attention_mask, dtype=torch.bool
                )
                record_trace(steps)
                return {
                    "sequences": xt,
                    "attention_mask": xt_attention_mask,
                    "history": history,
                    "attention_mask_history": attention_history,
                    "insertion_history": insertion_history,
                }
            return xt

        if confidence_method == "origin":
            xt = _origin_unmask_step(
                xt,
                unmask_logits,
                mask_indices,
                t=t,
                dt=dt,
                temperature=temperature,
                eps=eps,
                hazard_rate_schedule=unmasking_schedule,
                hazard_rate_exponent=unmasking_exponent,
                hazard_rate_params=unmasking_params,
                use_midpoint_hazard_rate=use_midpoint_hazard_rate,
            )
        else:
            xt = _adaptive_unmask_step(
                xt,
                unmask_logits,
                mask_indices,
                t=t,
                dt=dt,
                temperature=temperature,
                confidence_method=confidence_method,
                eps=eps,
                hazard_rate_schedule=unmasking_schedule,
                hazard_rate_exponent=unmasking_exponent,
                hazard_rate_params=unmasking_params,
                force_unmask_long_segments=force_unmask_long_segments,
                force_unmask_min_segment_len=force_unmask_min_segment_len,
                force_unmask_top_k_per_position=force_unmask_top_k_per_position,
                meaningful_replace=meaningful_replace,
                meaningful_replace_min_segment_len=meaningful_replace_min_segment_len,
                meaningful_replace_window_len=meaningful_replace_window_len,
                meaningful_replace_min_step=meaningful_replace_min_step,
                meaningful_token_mask=meaningful_token_mask,
                step_idx=step_idx,
                use_midpoint_hazard_rate=use_midpoint_hazard_rate,
            )

        xt, xt_attention_mask, last_is_inserted = _insertion_step(
            xt,
            xt_attention_mask,
            log_length_pred,
            prompt_len=prompt_len,
            t=t,
            dt=dt,
            pad_id=pad_id,
            mask_id=mask_id,
            eps=eps,
            hazard_rate_schedule=insertion_schedule,
            hazard_rate_exponent=insertion_exponent,
            hazard_rate_params=insertion_params,
            insertion_count_sampler=insertion_count_sampler,
            use_midpoint_hazard_rate=use_midpoint_hazard_rate,
        )

        t = (t + dt).clamp_max(1.0 - eps)
        if trace:
            record_trace(step_idx + 1)

    if trace:
        return {
            "sequences": xt,
            "attention_mask": xt_attention_mask,
            "history": history,
            "attention_mask_history": attention_history,
            "insertion_history": insertion_history,
        }
    return xt


__all__ = [
    "flexmdm_generate",
    "generate",
    "gumbel_sample",
]

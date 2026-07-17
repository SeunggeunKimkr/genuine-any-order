import math

import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature):
    """
    Sample from the Gumbel-Softmax distribution and optionally apply softmax.
    """
    if temperature == 0.0:
        return logits
    else:
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits / temperature + gumbel_noise

def _sample_lpmdm_slot(
    slot_scores: torch.Tensor,
    confidence: str,
    slot_temperature: float,
) -> torch.Tensor:
    slot_temperature = float(slot_temperature)
    if slot_temperature < 0.0:
        raise ValueError(
            f"sampling.slot_temperature must be non-negative, got {slot_temperature}."
        )

    chosen_slot = slot_scores.argmax(dim=1)
    if slot_temperature == 0.0 or confidence not in (
        "avg_log_prob",
        "log_prob_mean",
        "min_log_prob",
    ):
        return chosen_slot

    finite_rows = torch.isfinite(slot_scores).any(dim=1)
    if not finite_rows.any():
        return chosen_slot

    probs = F.softmax(slot_scores[finite_rows].float() / slot_temperature, dim=1)
    sampled_slot = torch.multinomial(probs, num_samples=1).squeeze(1)
    chosen_slot = chosen_slot.clone()
    chosen_slot[finite_rows] = sampled_slot.to(chosen_slot.dtype)
    return chosen_slot


def _topk_lpmdm_slots(
    slot_scores: torch.Tensor,
    k: int,
    confidence: str,
    slot_temperature: float,
):
    B, max_seg_num = slot_scores.shape
    k_eff = max(1, min(int(k), max_seg_num))
    slot_temperature = float(slot_temperature)
    if slot_temperature < 0.0:
        raise ValueError(
            f"sampling.slot_temperature must be non-negative, got {slot_temperature}."
        )

    if slot_temperature > 0.0 and confidence in (
        "avg_log_prob",
        "log_prob_mean",
        "min_log_prob",
    ):
        # Gumbel-top-k == sampling k slots without replacement from softmax(scores/T).
        scores = slot_scores.float()
        u = torch.rand_like(scores).clamp_min(1e-20)
        gumbel = -torch.log((-torch.log(u)).clamp_min(1e-20))
        perturbed = scores / slot_temperature + gumbel
        perturbed = torch.where(torch.isfinite(slot_scores), perturbed, scores)
    else:
        perturbed = slot_scores

    _, topk_idx = perturbed.topk(k=k_eff, dim=1)
    original_topk_scores = slot_scores.gather(1, topk_idx)
    valid_topk = torch.isfinite(original_topk_scores)

    chosen_mask = torch.zeros_like(slot_scores, dtype=torch.bool)
    val_b, val_k = valid_topk.nonzero(as_tuple=True)
    chosen_mask[val_b, topk_idx[val_b, val_k]] = True

    primary_slot = topk_idx[:, 0].clone()
    primary_slot[~valid_topk[:, 0]] = -1
    return chosen_mask, primary_slot

@torch.no_grad()
def arm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False):
    """
    Autoregressive sampling:
      - generates one token at a time, left-to-right
      - starts at the first mask_id position (prompt end)
      - uses gumbel-softmax + argmax (same style as mdm_sampling)
      - if eos_id is provided, stops per sequence at first EOS and fills the rest with EOS
      - uses KV caching when the underlying model is causal
    """
    temperature = sampling_cfg.temperature
    eos_id = getattr(sampling_cfg, "eos_id", None)

    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    # start position per row = first mask_id (prompt end)
    is_mask = (xt == mask_id)
    any_mask = is_mask.any(dim=1)
    start_pos = is_mask.float().argmax(dim=1)                 # 0 if no mask
    start_pos = torch.where(any_mask, start_pos, torch.full_like(start_pos, L))

    # nothing to generate
    if int(start_pos.min().item()) >= L:
        # if eos_id is set, still ensure "fill to L with eos" is trivially satisfied (nothing to do)
        return (xt, track_xt) if track else xt

    # track which sequences are finished (EOS generated)
    done = torch.zeros(B, dtype=torch.bool, device=xt.device) if eos_id is not None else None

    inner = model.module if hasattr(model, "module") else model
    use_cache_path = bool(getattr(inner.config, "causal", False))

    if use_cache_path:
        past_kvs = None
        position_offset = 0
        # Process positions left-to-right one token at a time so the cache can be reused.
        # logits at position `pos` predict the token at `pos+1`; we never need logits at L-1.
        for pos in range(L - 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                new_embeds = inner.emb(xt[:, pos : pos + 1])
                logits, past_kvs = inner.forward_step(
                    inputs_embeds=new_embeds,
                    past_kvs=past_kvs,
                    position_offset=position_offset,
                )
            position_offset += 1

            next_pos = pos + 1
            can_fill = (next_pos >= start_pos) & (xt[:, next_pos] == mask_id)
            if eos_id is not None:
                can_fill = can_fill & (~done)

            if can_fill.any():
                logits_step = logits[:, -1, :]
                noisy = gumbel_softmax(logits_step, temperature=temperature)
                next_tok = torch.argmax(noisy, dim=-1)
                xt[can_fill, next_pos] = next_tok[can_fill]
                if eos_id is not None:
                    done = done | (can_fill & (next_tok == eos_id))

            if track:
                track_xt.append(xt.clone().detach().cpu())

            if eos_id is not None and bool(done.all().item()):
                break
    else:
        for pos in range(L):
            # only generate for rows where:
            #  - pos is in the generation region
            #  - token is still mask
            #  - and (if eos_id) we haven't already produced EOS
            can_fill = (pos >= start_pos) & (xt[:, pos] == mask_id)
            if eos_id is not None:
                can_fill = can_fill & (~done)

            if not can_fill.any():
                continue

            # to predict token at `pos`, we use logits at `pos-1`
            src = pos - 1
            if src < 0:
                continue

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = model(xt)  # (B, L, V)

            logits_step = logits[:, src, :]  # predicts token at pos
            noisy = gumbel_softmax(logits_step, temperature=temperature)  # (B, V)
            next_tok = torch.argmax(noisy, dim=-1)                        # (B,)

            xt[can_fill, pos] = next_tok[can_fill]

            if eos_id is not None:
                # mark sequences as done if EOS was just produced at this position
                done = done | (can_fill & (next_tok == eos_id))

            if track:
                track_xt.append(xt.clone().detach().cpu())

    # If eos_id is set: for any sequence that has EOS in the generation region,
    # fill all later positions (up to length L) with EOS.
    if eos_id is not None:
        pos_idx = torch.arange(L, device=xt.device).unsqueeze(0)  # (1, L)
        gen_region = pos_idx >= start_pos.unsqueeze(1)            # (B, L)
        eos_in_region = (xt == eos_id) & gen_region               # (B, L)
        any_eos = eos_in_region.any(dim=1)                        # (B,)

        if any_eos.any():
            first_eos = eos_in_region.float().argmax(dim=1)       # (B,) (meaningful only where any_eos)
            # fill strictly after first_eos (and within gen_region) with eos_id
            fill_mask = any_eos.unsqueeze(1) & gen_region & (pos_idx > first_eos.unsqueeze(1))
            xt[fill_mask] = eos_id

            if track:
                # optional: record the post-fill final state once
                track_xt.append(xt.clone().detach().cpu())

    return (xt, track_xt) if track else xt

def _first_eos_valid_mask(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    is_eos = tokens == int(eos_id)
    has_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1)
    first_eos = torch.where(
        has_eos,
        first_eos,
        torch.full_like(first_eos, tokens.shape[1] - 1),
    )
    pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    return pos <= first_eos.unsqueeze(1)

def _encode_segments_for_planner(model, segments: torch.Tensor, eos_id: int) -> torch.Tensor:
    segment_pooling = str(getattr(model.encoder.config, "segment_pooling", "mean"))
    if segment_pooling == "mean":
        hidden = model.encoder(segments)
        pooled = hidden.mean(dim=1)
    elif segment_pooling == "eos_mean":
        valid = _first_eos_valid_mask(segments, int(eos_id))
        hidden = model.encoder(segments, attn_mask=valid[:, None, None, :])
        valid_f = valid.to(hidden.dtype)
        pooled = (hidden * valid_f.unsqueeze(-1)).sum(dim=1) / valid_f.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
    elif segment_pooling == "cls":
        cls_id = int(model.cls_token_id)
        cls_col = segments.new_full((segments.shape[0], 1), cls_id)
        cls_segments = torch.cat([cls_col, segments], dim=1)
        valid = _first_eos_valid_mask(segments, int(eos_id))
        cls_valid = torch.ones(
            valid.shape[0], 1, dtype=torch.bool, device=valid.device
        )
        valid_full = torch.cat([cls_valid, valid], dim=1)
        hidden = model.encoder(cls_segments, attn_mask=valid_full[:, None, None, :])
        pooled = hidden[:, 0]
    else:
        raise ValueError(
            "model.encoder.segment_pooling must be one of: mean, eos_mean, cls; "
            f"got {segment_pooling!r}."
        )
    return model.encoder_to_planner(pooled)

def _aggregate_lpmdm_candidate_metrics(
    candidate_mass: torch.Tensor,
    candidate_plogp: torch.Tensor,
    candidate_max_prob: torch.Tensor,
    candidate_batch: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab_size = candidate_mass.shape[1]
    mass = candidate_mass.new_zeros((batch_size, vocab_size))
    plogp = candidate_plogp.new_zeros((batch_size, vocab_size))
    max_prob = candidate_max_prob.new_zeros((batch_size, vocab_size))
    if candidate_mass.numel() > 0:
        mass.index_add_(0, candidate_batch, candidate_mass)
        plogp.index_add_(0, candidate_batch, candidate_plogp)
        for row_idx, sample_idx in enumerate(candidate_batch.tolist()):
            max_prob[sample_idx] = torch.maximum(
                max_prob[sample_idx],
                candidate_max_prob[row_idx],
            )

    entropy = mass.new_zeros(mass.shape)
    concentration = mass.new_zeros(mass.shape)
    positive = mass > 0
    if positive.any():
        entropy[positive] = (
            torch.log(mass[positive]) - plogp[positive] / mass[positive]
        )
        concentration[positive] = max_prob[positive] / mass[positive]
    return mass, entropy, concentration

def _valid_token_ids_per_row(tokens: torch.Tensor, eos_id: int) -> list[list[int]]:
    if tokens.numel() == 0:
        return []

    valid = _first_eos_valid_mask(tokens, int(eos_id))
    out = []
    for row_idx in range(tokens.shape[0]):
        token_ids = tokens[row_idx][valid[row_idx]]
        out.append([int(x) for x in torch.unique(token_ids).tolist()])
    return out

def _validate_embedding_token_ids(
    tokens: torch.Tensor,
    vocab_size: int,
    tensor_name: str,
):
    if tokens.numel() == 0:
        return

    min_id = int(tokens.min().item())
    max_id = int(tokens.max().item())
    if min_id < 0 or max_id >= int(vocab_size):
        raise ValueError(
            f"{tensor_name} contains token IDs outside the embedding vocabulary "
            f"[0, {int(vocab_size)}): min={min_id}, max={max_id}. "
            "This usually means the tokenizer, cached eval data, mask_id, or "
            "model vocab_size do not match."
        )

@torch.no_grad()
def _decode_segments_ar(
    model,
    planner_conditions: torch.Tensor,
    max_seg_len: int,
    eos_id: int,
    temperature: float,
    chunk_size: int,
    analysis: bool = False,
):
    decoded_chunks = []
    log_prob_chunks = []
    analysis_mass_chunks = []
    analysis_plogp_chunks = []
    analysis_max_prob_chunks = []
    chunk_size = max(1, int(chunk_size))
    decoder = model.decoder
    for start in range(0, planner_conditions.shape[0], chunk_size):
        end = min(start + chunk_size, planner_conditions.shape[0])
        conditions = planner_conditions[start:end]
        B = conditions.shape[0]
        cond_embed = conditions if conditions.dim() == 3 else conditions.unsqueeze(1)
        segment = torch.full(
            (B, max_seg_len),
            int(eos_id),
            dtype=torch.long,
            device=conditions.device,
        )
        token_log_probs = conditions.new_zeros((B, max_seg_len))
        done = torch.zeros(B, dtype=torch.bool, device=conditions.device)
        if analysis:
            vocab_size = decoder.config.vocab_size
            candidate_mass = conditions.new_zeros((B, vocab_size), dtype=torch.float32)
            candidate_plogp = conditions.new_zeros((B, vocab_size), dtype=torch.float32)
            candidate_max_prob = conditions.new_zeros((B, vocab_size), dtype=torch.float32)

        past_kvs = None
        position_offset = 0

        for token_pos in range(max_seg_len):
            can_decode = ~done
            if not can_decode.any():
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                if token_pos == 0:
                    new_embeds = cond_embed
                else:
                    new_embeds = decoder.emb(segment[:, token_pos - 1 : token_pos])
                logits, past_kvs = decoder.forward_step(
                    inputs_embeds=new_embeds,
                    past_kvs=past_kvs,
                    position_offset=position_offset,
                )
            position_offset += new_embeds.shape[1]

            logits_step = logits[:, -1, :]
            log_probs = F.log_softmax(logits_step.float(), dim=-1)
            if analysis:
                probs = log_probs.exp()
                candidate_mass[can_decode] = candidate_mass[can_decode] + probs[can_decode]
                candidate_plogp[can_decode] = (
                    candidate_plogp[can_decode] + probs[can_decode] * log_probs[can_decode]
                )
                candidate_max_prob[can_decode] = torch.maximum(
                    candidate_max_prob[can_decode],
                    probs[can_decode],
                )
            noisy = gumbel_softmax(logits_step, temperature=temperature)
            next_tok = torch.argmax(noisy, dim=-1)
            step_log_prob = log_probs.gather(1, next_tok.unsqueeze(1)).squeeze(1)

            segment[can_decode, token_pos] = next_tok[can_decode]
            token_log_probs[can_decode, token_pos] = step_log_prob[can_decode].to(token_log_probs.dtype)
            done = done | (can_decode & (next_tok == int(eos_id)))

        decoded_chunks.append(segment)
        log_prob_chunks.append(token_log_probs)
        if analysis:
            analysis_mass_chunks.append(candidate_mass)
            analysis_plogp_chunks.append(candidate_plogp)
            analysis_max_prob_chunks.append(candidate_max_prob)

    decoded = torch.cat(decoded_chunks, dim=0)
    token_log_probs = torch.cat(log_prob_chunks, dim=0)
    if not analysis:
        return decoded, token_log_probs

    return (
        decoded,
        token_log_probs,
        torch.cat(analysis_mass_chunks, dim=0),
        torch.cat(analysis_plogp_chunks, dim=0),
        torch.cat(analysis_max_prob_chunks, dim=0),
    )

@torch.no_grad()
def lpmdm_sampling(
    model,
    xt,
    mask_id,
    sampling_cfg,
    device: torch.device = None,
    track: bool = False,
    analysis: bool = False,
):
    """
    Masked latent sampling for LP-MDM.

    Returns only generated answer tokens with shape
    (B, max_seg_num * max_seg_len). Each iteration decodes all still-masked
    latent slots, commits the highest-scoring segment per sample, and feeds the
    committed segment back through the encoder.
    """
    model = model.module if hasattr(model, "module") else model
    temperature = sampling_cfg.temperature
    slot_temperature = float(getattr(sampling_cfg, "slot_temperature", 0.0))
    eos_id = getattr(sampling_cfg, "eos_id", None)
    if eos_id is None:
        raise ValueError("lpmdm_sampling requires sampling_cfg.eos_id.")

    if device is None:
        device = xt.device
    xt = xt.to(device).clone()

    B, max_prompt_len = xt.shape
    max_seg_num = getattr(sampling_cfg, "max_seg_num", None)
    if max_seg_num is None:
        max_seg_num = model.planner.config.max_position - max_prompt_len
    max_seg_num = int(max_seg_num)
    max_seg_len = int(getattr(sampling_cfg, "max_seg_len", 32))
    candidate_chunk_size = int(getattr(sampling_cfg, "candidate_chunk_size", 32))
    unmasking_num = int(getattr(sampling_cfg, "unmasking_num", 1))
    if max_seg_num <= 0:
        raise ValueError(f"max_seg_num must be positive, got {max_seg_num}.")
    if max_seg_len <= 0:
        raise ValueError(f"max_seg_len must be positive, got {max_seg_len}.")
    if unmasking_num <= 0:
        raise ValueError(f"unmasking_num must be positive, got {unmasking_num}.")

    planner_len = max_prompt_len + max_seg_num
    if planner_len > model.planner.config.max_position:
        raise ValueError(
            f"planner sequence length {planner_len} exceeds max_position "
            f"{model.planner.config.max_position}"
        )

    is_mask = xt == mask_id
    any_mask = is_mask.any(dim=1)
    prompt_len = is_mask.float().argmax(dim=1)
    prompt_len = torch.where(
        any_mask,
        prompt_len,
        torch.full_like(prompt_len, max_prompt_len),
    )

    prompt_aug = xt.new_full((B, planner_len), int(eos_id))
    pos = torch.arange(max_prompt_len, device=device).unsqueeze(0)
    real_prompt = pos < prompt_len.unsqueeze(1)
    prompt_aug[:, :max_prompt_len] = torch.where(
        real_prompt,
        xt,
        xt.new_full(xt.shape, int(eos_id)),
    )
    _validate_embedding_token_ids(
        prompt_aug,
        model.planner.prompt_emb.num_embeddings,
        "lpmdm_sampling prompt_aug",
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        planner_inputs = model.planner.prompt_emb(prompt_aug).clone()

    slot_offset = torch.arange(max_seg_num, device=device)
    slot_pos = prompt_len.unsqueeze(1) + slot_offset.unsqueeze(0)
    batch_slots = torch.arange(B, device=device).unsqueeze(1).expand(B, max_seg_num)
    mask_token = model.mask_token.to(dtype=planner_inputs.dtype).view(1, 1, -1)
    planner_inputs[batch_slots, slot_pos] = mask_token.expand(B, max_seg_num, -1)
    eos_slot_emb = model.planner.prompt_emb(
        xt.new_full((B, max_seg_num), int(eos_id))
    ).to(planner_inputs.dtype)

    masked_slots = torch.ones(B, max_seg_num, dtype=torch.bool, device=device)
    answer_tokens = xt.new_full((B, max_seg_num, max_seg_len), int(eos_id))
    batch_idx = torch.arange(B, device=device)
    if track:
        track_steps = []
    if analysis:
        analysis_steps = []
        vocab_size = model.decoder.config.vocab_size

    max_iters = math.ceil(max_seg_num / unmasking_num)
    for _ in range(max_iters):
        active_rows = masked_slots.any(dim=1)
        if not active_rows.any():
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            planner_out = model.planner(planner_inputs)

        slot_conditions = planner_out[batch_slots, slot_pos]
        cand_batch, cand_slot = masked_slots.nonzero(as_tuple=True)
        cand_conditions = slot_conditions[masked_slots]
        if analysis:
            (
                cand_tokens,
                cand_log_probs,
                cand_mass,
                cand_plogp,
                cand_max_prob,
            ) = _decode_segments_ar(
                model,
                cand_conditions,
                max_seg_len,
                int(eos_id),
                temperature,
                candidate_chunk_size,
                analysis=True,
            )
        else:
            cand_tokens, cand_log_probs = _decode_segments_ar(
                model,
                cand_conditions,
                max_seg_len,
                int(eos_id),
                temperature,
                candidate_chunk_size,
            )

        valid = _first_eos_valid_mask(cand_tokens, int(eos_id))
        valid_f = valid.to(cand_log_probs.dtype)
        valid_count = valid_f.sum(dim=1).clamp_min(1.0)
        cand_scores = (cand_log_probs * valid_f).sum(dim=1) / valid_count

        slot_scores = cand_scores.new_full((B, max_seg_num), float("-inf"))
        if sampling_cfg.confidence in ("avg_log_prob", "log_prob_mean"):
            slot_scores[cand_batch, cand_slot] = cand_scores
        elif sampling_cfg.confidence == "random":
            slot_scores[cand_batch, cand_slot] = torch.rand_like(cand_scores)
        elif sampling_cfg.confidence == "l2r":
            slot_scores[cand_batch, cand_slot] = -cand_slot.to(cand_scores.dtype)
        elif sampling_cfg.confidence == "min_log_prob":
            cand_scores = (cand_log_probs * valid_f).min(dim=1).values
            slot_scores[cand_batch, cand_slot] = cand_scores
        else:
            raise NotImplementedError(f"Confidence scoring strategy '{sampling_cfg.confidence}' not supported")

        chosen_slot_mask, primary_slot = _topk_lpmdm_slots(
            slot_scores,
            k=unmasking_num,
            confidence=sampling_cfg.confidence,
            slot_temperature=slot_temperature,
        )

        cand_tokens_bks = answer_tokens.new_full(
            (B, max_seg_num, max_seg_len),
            int(eos_id),
        )
        cand_tokens_bks[cand_batch, cand_slot] = cand_tokens
        if track:
            step_segments = answer_tokens.clone()
            step_segments[cand_batch, cand_slot] = cand_tokens
            chosen_slot_logged = primary_slot.to(dtype=torch.long)
        if analysis:
            step_mass, step_entropy, step_max_prob = _aggregate_lpmdm_candidate_metrics(
                cand_mass,
                cand_plogp,
                cand_max_prob,
                cand_batch,
                B,
            )

        chosen_batch, chosen_slot_idx = chosen_slot_mask.nonzero(as_tuple=True)
        chosen_tokens = cand_tokens_bks[chosen_batch, chosen_slot_idx]
        chosen_pos = prompt_len[chosen_batch] + chosen_slot_idx
        chosen_is_all_eos = (chosen_tokens == int(eos_id)).all(dim=1)
        non_eos_chosen = ~chosen_is_all_eos

        # Commit each chosen slot's tokens and drop it from the masked set.
        answer_tokens[chosen_batch, chosen_slot_idx] = chosen_tokens
        masked_slots[chosen_batch, chosen_slot_idx] = False

        if non_eos_chosen.any():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                chosen_latents = _encode_segments_for_planner(
                    model,
                    chosen_tokens[non_eos_chosen],
                    int(eos_id),
                ).to(planner_inputs.dtype)
            planner_inputs[
                chosen_batch[non_eos_chosen],
                chosen_pos[non_eos_chosen],
            ] = chosen_latents
        if chosen_is_all_eos.any():
            e_rows = chosen_batch[chosen_is_all_eos]
            e_slots = chosen_slot_idx[chosen_is_all_eos]
            planner_inputs[e_rows, prompt_len[e_rows] + e_slots] = eos_slot_emb[e_rows, e_slots]

        # Boundary: for each row, find the smallest chosen-slot index that was all-EOS.
        boundary_slot = torch.full((B,), max_seg_num, dtype=torch.long, device=device)
        if chosen_is_all_eos.any():
            e_rows = chosen_batch[chosen_is_all_eos]
            e_slots = chosen_slot_idx[chosen_is_all_eos]
            boundary_slot.scatter_reduce_(
                0,
                e_rows,
                e_slots.to(boundary_slot.dtype),
                reduce="amin",
                include_self=True,
            )

        has_boundary = boundary_slot < max_seg_num
        if has_boundary.any():
            boundary_rows = has_boundary.nonzero(as_tuple=True)[0]
            row_boundary = boundary_slot[boundary_rows]
            tail_mask = slot_offset.unsqueeze(0) >= row_boundary.unsqueeze(1)
            eos_segs = answer_tokens.new_full(
                (boundary_rows.numel(), max_seg_num, max_seg_len),
                int(eos_id),
            )
            answer_tokens[boundary_rows] = torch.where(
                tail_mask.unsqueeze(-1),
                eos_segs,
                answer_tokens[boundary_rows],
            )
            masked_slots[boundary_rows] = masked_slots[boundary_rows] & ~tail_mask
            # Overwrite planner_inputs for every slot in the tail (whether it was
            # chosen this step with non-EOS content or committed in an earlier step).
            t_batch_exp = boundary_rows.unsqueeze(1).expand(-1, max_seg_num)
            t_slot_exp = slot_offset.unsqueeze(0).expand(boundary_rows.numel(), -1)
            t_rows = t_batch_exp[tail_mask]
            t_slots = t_slot_exp[tail_mask]
            planner_inputs[t_rows, prompt_len[t_rows] + t_slots] = eos_slot_emb[t_rows, t_slots]

        if analysis:
            highlight_token_ids = [[] for _ in range(B)]
            chosen_valid_ids = _valid_token_ids_per_row(chosen_tokens, int(eos_id))
            for row_idx, sample_idx in enumerate(chosen_batch.tolist()):
                highlight_token_ids[sample_idx] = sorted(
                    set(highlight_token_ids[sample_idx]) | set(chosen_valid_ids[row_idx])
                )
            analysis_steps.append(
                {
                    "mass": step_mass.detach().cpu(),
                    "entropy": step_entropy.detach().cpu(),
                    "max_prob": step_max_prob.detach().cpu(),
                    "highlight_token_ids": highlight_token_ids,
                    "has_valid_positions": active_rows.detach().cpu(),
                }
            )
        if track:
            track_steps.append(
                {
                    "segments": step_segments.clone().detach().cpu(),
                    "slot_scores": slot_scores.clone().detach().cpu(),
                    "answer_tokens": answer_tokens.clone().detach().cpu(),
                    "masked_slots": masked_slots.clone().detach().cpu(),
                    "chosen_slot": chosen_slot_logged.clone().detach().cpu(),
                }
            )

    if analysis:
        empty_mass = torch.zeros((B, vocab_size), dtype=torch.float32)
        empty_entropy = torch.zeros((B, vocab_size), dtype=torch.float32)
        empty_max_prob = torch.zeros((B, vocab_size), dtype=torch.float32)
        empty_valid = torch.zeros(B, dtype=torch.bool)
        while len(analysis_steps) < max_seg_num:
            analysis_steps.append(
                {
                    "mass": empty_mass.clone(),
                    "entropy": empty_entropy.clone(),
                    "max_prob": empty_max_prob.clone(),
                    "highlight_token_ids": [[] for _ in range(B)],
                    "has_valid_positions": empty_valid.clone(),
                }
            )
    if track:
        while len(track_steps) < max_seg_num:
            track_steps.append(
                {
                    "segments": answer_tokens.clone().detach().cpu(),
                    "slot_scores": answer_tokens.new_full(
                        (B, max_seg_num),
                        float("-inf"),
                        dtype=torch.float,
                    ).cpu(),
                    "answer_tokens": answer_tokens.clone().detach().cpu(),
                    "masked_slots": masked_slots.clone().detach().cpu(),
                    "chosen_slot": torch.full(
                        (B,),
                        -1,
                        dtype=torch.long,
                        device=answer_tokens.device,
                    ).cpu(),
                }
            )

    answer_tokens = answer_tokens.view(B, max_seg_num * max_seg_len)
    if track and analysis:
        return answer_tokens, track_steps, analysis_steps
    if track:
        return answer_tokens, track_steps
    if analysis:
        return answer_tokens, analysis_steps
    return answer_tokens

def _mdm_prompt_len(tokens: torch.Tensor, mask_id: int) -> torch.Tensor:
    is_mask = tokens == int(mask_id)
    any_mask = is_mask.any(dim=1)
    prompt_len = is_mask.float().argmax(dim=1)
    return torch.where(
        any_mask,
        prompt_len,
        torch.full_like(prompt_len, tokens.shape[1]),
    )


def _fill_mdm_after_first_eos(
    tokens: torch.Tensor,
    eos_id: int,
    prompt_len: torch.Tensor,
) -> torch.Tensor:
    if eos_id is None or tokens.numel() == 0:
        return tokens

    B, L = tokens.shape
    prompt_len = prompt_len.to(device=tokens.device, dtype=torch.long)
    pos = torch.arange(L, device=tokens.device).unsqueeze(0)
    gen_region = pos >= prompt_len.unsqueeze(1)
    eos_revealed = (tokens == int(eos_id)) & gen_region
    has_eos = eos_revealed.any(dim=1)
    if not has_eos.any():
        return tokens

    first_eos = eos_revealed.float().argmax(dim=1)
    fill_mask = has_eos.unsqueeze(1) & gen_region & (pos > first_eos.unsqueeze(1))
    return torch.where(fill_mask, tokens.new_full((), int(eos_id)), tokens)


@torch.no_grad()
def mdm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False, arm_init: bool = False):
    # sampling hyperparameters
    # xt can include clean tokens
    # if track == True, we return the trace (used for the debugging purpose)
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    generate_until = bool(getattr(sampling_cfg, "generate_until", False))
    eos_id = getattr(sampling_cfg, "eos_id", None)
    if generate_until and eos_id is None:
        raise ValueError("mdm_sampling requires sampling_cfg.eos_id when generate_until=true.")

    # shape
    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    if arm_init:
        xt_t1, xt = xt[:, :1], xt[:, 1:]
        L = L - 1

    if generate_until:
        mdm_prompt_len = _mdm_prompt_len(xt, mask_id)
        xt = _fill_mdm_after_first_eos(xt, int(eos_id), mdm_prompt_len)

    for i in range(L // unmasking_num + 1):
        # mask indicies
        mask_indices = (xt == mask_id)

        if mask_indices.sum() == 0:
            break

        # calculate logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
            logits = model(torch.cat([xt_t1, xt], dim=1) if arm_init else xt) # [B, L, V]
        if arm_init:
            logits = logits[:, :-1, :]
        logits_with_noise = gumbel_softmax(logits, temperature = temperature)
        p = F.softmax(logits, dim = -1)

        if confidence == "top_k":
            unmasking_score = torch.where(mask_indices, p.max(dim = -1).values, -float('inf'))
        elif confidence == "top_k_margin":
            probs_top_2 = p.topk(k=2, dim=-1).values
            unmasking_score = torch.where(mask_indices, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
        elif confidence == "entropy":
            entropy = (- p * torch.log(p + 1e-10)).sum(dim = -1)
            unmasking_score = torch.where(mask_indices, entropy, -float('inf'))
        elif confidence == "random":
            unmasking_score = torch.where(mask_indices, torch.rand_like(p[:, :, 0]), -float('inf'))
        elif confidence == "l2r":
            positions = torch.arange(xt.shape[1], device=xt.device, dtype=p.dtype).unsqueeze(0)
            unmasking_score = torch.where(mask_indices, -positions, -float('inf'))
        else:
            raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")

        # update masked tokens by selecting top-k per batch this step
        for j in range(B):
            k = min(unmasking_num, int(mask_indices[j].sum().item())) # number of tokens to unmask
            if k > 0:
                _, select_indices = torch.topk(unmasking_score[j], k=k)
                xt[j, select_indices] = torch.argmax(logits_with_noise[j, select_indices], dim = -1)

        if generate_until:
            xt = _fill_mdm_after_first_eos(xt, int(eos_id), mdm_prompt_len)

        if track:
            cur = torch.cat([xt_t1, xt], dim=1) if arm_init else xt
            track_xt.append(cur.clone().detach().cpu())
    if arm_init:
        xt = torch.cat([xt_t1, xt], dim=1)
    if track:
        return xt, track_xt
    else:
        return xt

@torch.no_grad()
def mdm_sampling_block(model, xt, block_size, mask_id, sampling_cfg, device: torch.device = None):
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    device = xt.device

    # shape 
    B, L = xt.shape
    assert L % block_size == 0, "block size must be divisible by the max_length"
    n_blocks = L // block_size
    xt = xt.clone()

    for n in range(n_blocks):
        s = n * block_size
        e = s + block_size

        for i in range(block_size // unmasking_num + 2):
            valid_mask_ids = (xt[:, s:e] == mask_id) # [B, e]
            if valid_mask_ids.sum() == 0:
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                logits = model(xt[:, : e]) # [B, e, V]

            logits_with_noise = gumbel_softmax(logits[:, s:e, :], temperature = temperature) # [B, e, V]
            p = F.softmax(logits[:, s:e, :], dim = -1)

            if confidence == "top_k":
                unmasking_score = torch.where(valid_mask_ids, p.max(dim = -1).values, -float('inf'))
            elif confidence == "top_k_margin":
                probs_top_2 = p.topk(k=2, dim=-1).values
                unmasking_score = torch.where(valid_mask_ids, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
            elif confidence == "random":
                raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
            else:
                raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
                
            for j in range(B):
                k = min(unmasking_num, int(valid_mask_ids[j].sum().item())) # number of tokens to unmask
                if k > 0:
                    _, select_indices = torch.topk(unmasking_score[j], k=k)
                    xt[j, s + select_indices] = torch.argmax(logits_with_noise[j, select_indices], dim = -1)
    
    assert (xt == mask_id).sum() == 0, "There are still masked tokens in the input"
            
    return xt

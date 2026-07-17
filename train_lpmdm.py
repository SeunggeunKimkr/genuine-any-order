import math, os, time, json, random, sys, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributed as dist
import argparse
from contextlib import nullcontext
from copy import deepcopy
from types import MethodType
from tqdm import tqdm
from model.transformer import MDMTransformer, MDMConfig, CombinedTransformer
from data import setup_data_bundle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig
from model.ema import ExponentialMovingAverage, save_model_snapshot
from eval.gsm8k_eval import evaluate_ddp_gsm8k

COMBINED_STRATEGIES = {"lpmdm"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    # Extras are OmegaConf dotlist overrides, e.g. model.planner.apply_final_norm=false
    args, extras = parser.parse_known_args()
    args.overrides = extras
    return args


def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=1),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank

def _is_sequence_config(value):
    return isinstance(value, (list, tuple, ListConfig))

def _as_list(value):
    if _is_sequence_config(value):
        return list(value)
    return [value]

def _add_eval_result(out, prefix, result):
    if isinstance(result, dict):
        for key, value in result.items():
            out[f"{prefix}_{key}"] = value
    else:
        out[prefix] = result

def evaluate_ddp_dict(model, cfg, device, rank, world_size):
    sampling = cfg.validation.sampling
    strategy = cfg.training.strategy

    if strategy == "lpmdm":
        confidence_cfg = getattr(sampling, "confidence", None)
        unmasking_num_cfg = getattr(sampling, "unmasking_num", 1)
        confidence_is_list = _is_sequence_config(confidence_cfg)
        unmasking_num_is_list = _is_sequence_config(unmasking_num_cfg)
        confidence_values = (
            _as_list(confidence_cfg) if confidence_cfg is not None else [None]
        )
        unmasking_values = _as_list(unmasking_num_cfg)

        out = {}
        for confidence_value in confidence_values:
            for unmasking_num_value in unmasking_values:
                sampling_i = deepcopy(sampling)
                if confidence_value is not None:
                    sampling_i.confidence = confidence_value
                sampling_i.unmasking_num = int(unmasking_num_value)
                result = evaluate_ddp(model, cfg, device, rank, world_size, sampling_i)
                key_parts = [strategy]
                if confidence_is_list and confidence_value is not None:
                    key_parts.append(str(confidence_value))
                if unmasking_num_is_list:
                    key_parts.append(f"unmasking_{int(unmasking_num_value)}")
                key = "_".join(key_parts)
                _add_eval_result(out, key, result)
        return out

    if strategy == "arm":
        confidence = getattr(sampling, "confidence", None)
        if confidence is None:
            result = evaluate_ddp(model, cfg, device, rank, world_size, sampling)
            out = {}
            _add_eval_result(out, strategy, result)
            return out

        out = {}
        confidence_is_list = _is_sequence_config(confidence)
        for confidence_value in _as_list(confidence):
            sampling_i = deepcopy(sampling)
            sampling_i.confidence = confidence_value
            result = evaluate_ddp(model, cfg, device, rank, world_size, sampling_i)
            key = (
                f"{strategy}_{confidence_value}"
                if confidence_is_list
                else strategy
            )
            _add_eval_result(out, key, result)
        return out

    base_sampling = sampling
    out = {}

    for confidence in _as_list(base_sampling.confidence):
        for unmasking_num in _as_list(base_sampling.unmasking_num):
            sampling = deepcopy(base_sampling)
            sampling.confidence = confidence
            sampling.unmasking_num = unmasking_num
            result = evaluate_ddp(model, cfg, device, rank, world_size, sampling)
            _add_eval_result(out, f"{confidence}_unmasking_{unmasking_num}", result)
    return out

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(p=2).item()
            total += param_norm ** 2
    return total ** 0.5

def evaluate_ddp(model, cfg, device, rank: int, world_size: int, sampling):
    if cfg.data.dataset in ("tinygsm", "tinygsm_split_v3", "tinygsm_split_v2"):
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

# mdm loss implementation
def mdm_loss(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False):
    # sample integer uniformly for each batch from [1,L]
    # prompt_mask (boolean mask): 1 for prompt
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1 , keepdim=True)
    # uniformly sample the number of positions to mask
    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    # mask correspondent number of tokens for each batch, 0.0 for the prompt indices
    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    # calculate (reweighted) loss
    num_mask = num_mask.float().expand_as(mask_indices)

    if arm_init:
        ce = F.cross_entropy(logits[:, :-1, :][mask_indices[:, 1:]], input_ids[:, 1:][mask_indices[:, 1:]], reduction="none")
    else:
        ce = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction="none")
    loss = ce / num_mask[mask_indices]
    return loss.sum() / B

def arm_loss(
    model,
    input_ids: torch.Tensor,                    # (B, L)
    eos_id: int,
    prompt_mask: Optional[torch.Tensor] = None, # True = prompt token
):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids)          # (B, L, V)
    targets = input_ids[:, 1:]         # (B, L-1)
    pred_logits = logits[:, :-1, :]    # (B, L-1, V)

    valid = ~prompt_mask[:, 1:]        # (B, L-1)

    if eos_id is not None:
        is_eos = (targets == eos_id)               # (B, L-1)
    else:
        is_eos = torch.zeros_like(targets, dtype=torch.bool)
    any_eos = is_eos.any(dim=1)                # (B,)
    first_eos = is_eos.float().argmax(dim=1)   # (B,) 0-based in targets
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, targets.shape[1] - 1),
    )

    t = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)  # (1, L-1)
    valid = valid & (t <= first_eos.unsqueeze(1))

    if valid.sum().item() == 0:
        return pred_logits.sum() * 0.0
    return F.cross_entropy(pred_logits[valid], targets[valid], reduction="mean")

def lpmdm_next_token_valid_mask(target_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    is_eos = target_ids == eos_id
    any_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1)
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, target_ids.shape[1] - 1),
    )
    t = torch.arange(target_ids.shape[1], device=target_ids.device).unsqueeze(0)
    return t <= first_eos.unsqueeze(1)

def encode_segments_for_planner(
    model,
    segments: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    segment_pooling = str(getattr(model.encoder.config, "segment_pooling", "mean"))
    if segment_pooling == "mean":
        hidden = model.encoder(segments)
        pooled = hidden.mean(dim=1)
    elif segment_pooling == "eos_mean":
        valid = lpmdm_next_token_valid_mask(segments, int(eos_id))
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
        valid = lpmdm_next_token_valid_mask(segments, int(eos_id))
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

def lpmdm_decoder_ce_loss(
    model,
    planner_conditions: torch.Tensor,
    target_ids: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int,
    slot_weights: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    valid = lpmdm_next_token_valid_mask(target_ids, eos_id)
    total_loss = planner_conditions.new_zeros(())
    if slot_weights is None:
        total_count = planner_conditions.new_zeros(())
    else:
        slot_weights = slot_weights.to(
            device=planner_conditions.device,
            dtype=total_loss.dtype,
        )
        if slot_weights.shape[0] != target_ids.shape[0]:
            raise ValueError(
                "slot_weights must have one weight per decoder target segment."
            )
        if batch_size is None:
            raise ValueError("batch_size is required when slot_weights is set.")
        batch_count = total_loss.new_tensor(float(batch_size)).clamp(min=1.0)

    for start in range(0, target_ids.shape[0], decoder_chunk_size):
        end = min(start + decoder_chunk_size, target_ids.shape[0])
        chunk_targets = target_ids[start:end]
        chunk_valid = valid[start:end]
        if not chunk_valid.any():
            continue
        logits = model.decoder(chunk_targets, planner_conditions[start:end])
        pred_logits = logits[:, :-1, :]
        token_loss = F.cross_entropy(
            pred_logits[chunk_valid],
            chunk_targets[chunk_valid],
            reduction="none" if slot_weights is not None else "sum",
        )
        if slot_weights is None:
            total_loss = total_loss + token_loss
            total_count = total_count + chunk_valid.sum().to(total_loss.dtype)
        else:
            valid_slot_idx = chunk_valid.nonzero(as_tuple=True)[0]
            chunk_slot_loss = total_loss.new_zeros((end - start,))
            chunk_slot_loss.index_add_(
                0,
                valid_slot_idx,
                token_loss.to(total_loss.dtype),
            )
            total_loss = total_loss + (
                chunk_slot_loss * slot_weights[start:end]
            ).sum()

    if slot_weights is None:
        return total_loss / total_count.clamp(min=1.0)
    return total_loss / batch_count

def lpmdm_forward(
    model,
    prompt_ids: torch.Tensor,
    split_labels: torch.Tensor,
    prompt_len: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int = 1024,
    slot_reweight: bool = False,
) -> torch.Tensor:
    B, max_prompt_len = prompt_ids.shape
    _, max_seg_num, max_seg_len = split_labels.shape
    planner_len = max_prompt_len + max_seg_num
    if planner_len > model.planner.config.max_position:
        raise ValueError(
            f"planner sequence length {planner_len} exceeds max_position "
            f"{model.planner.config.max_position}"
        )

    device = prompt_ids.device
    prompt_len = prompt_len.to(device=device, dtype=torch.long).clamp(0, max_prompt_len)

    prompt_aug = prompt_ids.new_full((B, planner_len), int(eos_id))
    prompt_aug[:, :max_prompt_len] = prompt_ids
    planner_inputs = model.planner.prompt_emb(prompt_aug).clone()

    segment_is_real = ~(split_labels == int(eos_id)).all(dim=2)
    if segment_is_real.any():
        flat_segments = split_labels[segment_is_real]
        flat_latents = encode_segments_for_planner(
            model,
            flat_segments,
            int(eos_id),
        ).to(planner_inputs.dtype)
        batch_idx, seg_idx = segment_is_real.nonzero(as_tuple=True)
        slot_pos = prompt_len[batch_idx] + seg_idx
        planner_inputs[batch_idx, slot_pos] = flat_latents

    num_mask = (
        torch.floor(torch.rand(B, 1, device=device) * max_seg_num).long() + 1
    )
    segment_scores = torch.rand((B, max_seg_num), device=device)
    segment_order = segment_scores.argsort(dim=1).argsort(dim=1)
    segment_mask = segment_order < num_mask

    slot_pos = prompt_len.unsqueeze(1) + torch.arange(max_seg_num, device=device).unsqueeze(0)
    batch_slots = torch.arange(B, device=device).unsqueeze(1).expand(B, max_seg_num)
    mask_indices = torch.zeros((B, planner_len), dtype=torch.bool, device=device)
    mask_indices[batch_slots, slot_pos] = segment_mask

    mask_token = model.mask_token.to(dtype=planner_inputs.dtype).view(1, 1, -1)
    planner_inputs = torch.where(mask_indices.unsqueeze(-1), mask_token, planner_inputs)
    planner_out = model.planner(planner_inputs)

    masked_batch, masked_pos = mask_indices.nonzero(as_tuple=True)
    if masked_batch.numel() == 0:
        return planner_out.sum() * 0.0

    planner_conditions = planner_out[masked_batch, masked_pos]
    tail_idx = masked_pos - prompt_len[masked_batch]

    target_ids = split_labels[masked_batch, tail_idx]
    slot_weights = None
    if slot_reweight:
        num_mask_per_sample = num_mask.squeeze(1).to(dtype=planner_conditions.dtype)
        slot_weights = 1.0 / num_mask_per_sample[masked_batch]

    return lpmdm_decoder_ce_loss(
        model,
        planner_conditions,
        target_ids,
        int(eos_id),
        max(1, int(decoder_chunk_size)),
        slot_weights=slot_weights,
        batch_size=B if slot_reweight else None,
    )

def attach_lpmdm_forward(model):
    model.forward = MethodType(lpmdm_forward, model)
    return model

def lpmdm_loss(
    model,
    prompt: torch.Tensor,
    split_labels: torch.Tensor,
    prompt_len: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int = 1024,
    slot_reweight: bool = False,
):
    return model(
        prompt,
        split_labels,
        prompt_len,
        eos_id=int(eos_id),
        decoder_chunk_size=int(decoder_chunk_size),
        slot_reweight=bool(slot_reweight),
    )

def get_prompt_len(batch, device):
    if "prompt_mask" in batch:
        return batch["prompt_mask"].to(device).sum(dim=1).long()
    return batch["prompt_len"].to(device)

# validation loss helper
def val_loss_ddp(
    model,
    val_loader,
    mask_id: int,
    device,
    rank: int,
    world_size: int,
    strategy: str,
    eos_id: int,
    arm_init: bool = False,
    decoder_chunk_size: int = 1024,
    slot_reweight: bool = False,
):
    model.eval()
    if world_size > 1 and dist.is_initialized() and not isinstance(val_loader.sampler, DistributedSampler):
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size or 16,
            sampler=sampler,
            # Avoid forking validation workers after CUDA has been initialized.
            # Forked workers inherit CUDA tensors from the rank process and can
            # abort during teardown with "CUDA error: initialization error".
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
    else:
        sampler = None

    local_sum = 0.0
    local_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc = "Validating", disable = (rank != 0)):
            # to enable flashattention, we do autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy in COMBINED_STRATEGIES:
                    prompt = batch["prompt"].to(device)
                    split_labels = batch["split_labels"].to(device)
                    prompt_len = get_prompt_len(batch, device)
                    loss = lpmdm_loss(
                        model,
                        prompt,
                        split_labels,
                        prompt_len,
                        eos_id=eos_id,
                        decoder_chunk_size=decoder_chunk_size,
                        slot_reweight=slot_reweight,
                    )
                    B = prompt.shape[0]
                elif strategy == "arm":
                    x0 = batch["labels"].to(device)
                    pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    B = x0.shape[0]
                    loss = arm_loss(model, x0, eos_id=eos_id, prompt_mask=pm)
                elif strategy == "standard":
                    x0 = batch["labels"].to(device)
                    pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                    B = x0.shape[0]
                    loss = mdm_loss(model, x0, mask_id, prompt_mask = pm, arm_init=arm_init)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            local_sum += float(loss.item() * B)
            local_count += B
    
    tensor = torch.tensor([local_sum, local_count], dtype=torch.float, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count = tensor.tolist()

    return global_sum / max(int(global_count), 1)

def main(cfg: DictConfig, resume_path: str = None):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")
    
    base_seed = 2026
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ckpt dir
    ckpt_dir = f"ckpts/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}-{random.SystemRandom().randint(0, 99999):05d}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    data_cfg = cfg.data
    train_cfg = cfg.training
    val_cfg = cfg.validation
    strategy = train_cfg.strategy

    # Initialize the model
    model_cfg_dict = cfg.model
    if strategy in COMBINED_STRATEGIES:
        encoder_config = MDMConfig(**model_cfg_dict.encoder)
        planner_config = MDMConfig(**model_cfg_dict.planner)
        decoder_config = MDMConfig(**model_cfg_dict.decoder)
        model_config = planner_config
        model = CombinedTransformer(
            encoder_config,
            planner_config,
            decoder_config,
            tie_embeddings=bool(model_cfg_dict.get("tie_embeddings", False)),
        ).to(device)
        attach_lpmdm_forward(model)
    else:
        model_config = MDMConfig(**model_cfg_dict)
        model = MDMTransformer(model_config).to(device)

    # ARM initialization
    arm_init_path = model_cfg_dict.get("arm_init", "none") if strategy not in COMBINED_STRATEGIES else "none"
    if arm_init_path != "none":
        model_config.predict_next_token = True
        if is_main:
            print(f"Initializing MDM from ARM checkpoint: {arm_init_path}")
        arm_ckpt = torch.load(arm_init_path, map_location="cpu")
        sd = arm_ckpt.get("model_state_dict", arm_ckpt)
        model.load_state_dict(sd, strict=True)


    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main:
            print(f"Model wrapping is done!")

    # data
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"
    data_bundle = setup_data_bundle(data_cfg)
    train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id
    eos_id = getattr(val_cfg.sampling, "eos_id", None)
    if eos_id is None:
        eos_id = getattr(train_cfg, "eos_id", None)
    if strategy in COMBINED_STRATEGIES and eos_id is None:
        raise ValueError("LP-MDM training requires validation.sampling.eos_id or training.eos_id.")
    decoder_chunk_size = int(getattr(train_cfg, "decoder_chunk_size", 1024))
    slot_reweight = bool(getattr(train_cfg, "slot_reweight", False))
    gradient_accumulation_steps = int(getattr(train_cfg, "gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1.")
    if is_main and gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation: {gradient_accumulation_steps} microbatches per optimizer step")

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    num_training_steps = train_cfg.num_epochs * num_update_steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    # resume from checkpoint
    start_epoch = 0
    start_global_step = 0
    if resume_path is not None:
        if is_main:
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if train_cfg.ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.move_shadow_params_to_device(device)
        start_epoch = ckpt.get("epoch", 0)
        start_global_step = ckpt.get("global_step", 0)
        if is_main:
            print(f"Resumed at epoch {start_epoch}, global_step {start_global_step}")

    # training loop
    global_step = start_global_step
    last_grad_norm = None
    optimizer.zero_grad(set_to_none=True)

        
    # wandb initialize
    if cfg.wandb.wandb and is_main:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)

    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        # compute how many steps to skip in this epoch when resuming
        steps_in_epoch = len(train_loader)
        skip_steps = (
            max(0, start_global_step * gradient_accumulation_steps - epoch * steps_in_epoch)
            if epoch == start_epoch
            else 0
        )
        accum_count = 0
        accum_window_size = gradient_accumulation_steps

        for step_in_epoch, itr in enumerate(pbar):
            if step_in_epoch < skip_steps:
                continue
            if accum_count == 0:
                remaining_micro_steps = steps_in_epoch - step_in_epoch
                accum_window_size = min(gradient_accumulation_steps, remaining_micro_steps)
            is_update_step = (accum_count + 1) == accum_window_size

            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and not is_update_step
                else nullcontext()
            )
            with sync_context:
                # to enable flashattention, we do the autocast
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                    if strategy == "standard":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                        loss = mdm_loss(model, input_ids, mask_id, prompt_mask = prompt_mask, arm_init=model_config.predict_next_token)
                    elif strategy == "arm":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                        loss = arm_loss(model, input_ids, eos_id=eos_id, prompt_mask=prompt_mask)
                    elif strategy in COMBINED_STRATEGIES:
                        batch = itr
                        prompt = batch["prompt"].to(device)
                        split_labels = batch["split_labels"].to(device)
                        prompt_len = get_prompt_len(batch, device)
                        loss = lpmdm_loss(
                            model,
                            prompt,
                            split_labels,
                            prompt_len,
                            eos_id=eos_id,
                            decoder_chunk_size=decoder_chunk_size,
                            slot_reweight=slot_reweight,
                        )
                    else:
                        raise ValueError(f"Invalid training strategy: {strategy}")

                (loss / accum_window_size).backward()
            accum_count += 1

            if is_update_step:
                if train_cfg.max_grad_norm > 0:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_cfg.max_grad_norm,
                    ).item()
                else:
                    last_grad_norm = grad_norm(model.parameters())
                optimizer.step()
                if train_cfg.ema is not None:
                    ema.update(ema_params)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_count = 0

            if is_main:
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

                if is_update_step and global_step % train_cfg.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss {loss.item()}")
                    if cfg.wandb.wandb:
                        wandb.log({"loss": loss.item()}, step=global_step)

                        if last_grad_norm is not None:
                            wandb.log({"grad_norm": last_grad_norm}, step=global_step)

            if is_update_step and global_step % train_cfg.eval_steps == 0:
                model.eval()

                # validaton on the downstream task; disabled when we use EMA
                if train_cfg.ema is None:
                    val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                else:
                    val_acc_dict = None

                # validation loss (mdm loss on the validation dataset)
                val_loss = val_loss_ddp(
                    model,
                    val_loader,
                    mask_id,
                    device,
                    rank,
                    world_size,
                    strategy,
                    eos_id,
                    arm_init=model_config.predict_next_token,
                    decoder_chunk_size=decoder_chunk_size,
                    slot_reweight=slot_reweight,
                )

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())
                   
                    with torch.inference_mode():
                        # validaton on the downstream task
                        val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)
                    ema.restore(model_to_ema.parameters())
                
                if is_main:
                    # eval acc logging
                    for key, value in val_acc_dict.items():
                        print(f"Epoch {epoch+1}, Step {global_step}, Validation Accuracy {key}: {value}")
                        if cfg.wandb.wandb:
                            if train_cfg.ema is not None:
                                wandb.log({"ema_val_acc_" + key: value}, step=global_step)
                            else:
                                wandb.log({"val_acc_" + key: value}, step=global_step)
                    
                    # validation loss logging
                    print(f"Epoch {epoch+1}, Step {global_step}, Validation Loss: {val_loss}")
                    if cfg.wandb.wandb:
                        wandb.log({"val_loss": val_loss}, step=global_step)

                    if is_main and global_step % train_cfg.save_steps == 0:
                        # save non-EMA snapshot
                        save_extra = {
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        }
                        if train_cfg.ema is not None:
                            save_extra["ema_state_dict"] = ema.state_dict()
                        if val_acc_dict is not None:
                            save_extra.update(val_acc_dict)
                        saved_path = save_model_snapshot(
                            ckpt_dir, model, cfg, epoch, global_step,
                            val_loss=val_loss,
                            extra=save_extra,
                        )
                        if saved_path is not None:
                            print(f"Model saved to: {saved_path}")
                
                model.train()
    
    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg
    cfg = OmegaConf.load(cfg_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg, resume_path=args.resume)

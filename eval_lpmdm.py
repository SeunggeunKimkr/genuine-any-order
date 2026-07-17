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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to evaluate")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
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
    if cfg.data.dataset in ("tinygsm", "tinygsm_split_v2"):
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

def lpmdm_decoder_ce_loss(
    model,
    planner_conditions: torch.Tensor,
    target_ids: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int,
) -> torch.Tensor:
    valid = lpmdm_next_token_valid_mask(target_ids, eos_id)
    total_loss = planner_conditions.new_zeros(())
    total_count = planner_conditions.new_zeros(())

    for start in range(0, target_ids.shape[0], decoder_chunk_size):
        end = min(start + decoder_chunk_size, target_ids.shape[0])
        chunk_targets = target_ids[start:end]
        chunk_valid = valid[start:end]
        if not chunk_valid.any():
            continue
        logits = model.decoder(chunk_targets, planner_conditions[start:end])
        pred_logits = logits[:, :-1, :]
        total_loss = total_loss + F.cross_entropy(
            pred_logits[chunk_valid],
            chunk_targets[chunk_valid],
            reduction="sum",
        )
        total_count = total_count + chunk_valid.sum().to(total_loss.dtype)

    return total_loss / total_count.clamp(min=1.0)

def lpmdm_forward(
    model,
    prompt_ids: torch.Tensor,
    split_labels: torch.Tensor,
    prompt_len: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int = 1024,
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
        hidden = model.encoder(flat_segments)
        flat_latents = model.encoder_to_planner(hidden.mean(dim=1)).to(planner_inputs.dtype)
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

    return lpmdm_decoder_ce_loss(
        model,
        planner_conditions,
        target_ids,
        int(eos_id),
        max(1, int(decoder_chunk_size)),
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
):
    return model(
        prompt,
        split_labels,
        prompt_len,
        eos_id=int(eos_id),
        decoder_chunk_size=int(decoder_chunk_size),
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
    if resume_path is None:
        raise ValueError("eval_lpmdm.py requires --resume to point to a checkpoint.")

    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start evaluation!")
        print(f"Evaluating with {world_size} GPUs")
    
    base_seed = 2026
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    train_cfg = cfg.training
    val_cfg = cfg.validation
    strategy = train_cfg.strategy
    wandb_initialized = False

    try:
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
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            if is_main:
                print(f"Model wrapping is done!")

        eos_id = getattr(val_cfg.sampling, "eos_id", None)
        if eos_id is None:
            eos_id = getattr(train_cfg, "eos_id", None)
        if strategy in COMBINED_STRATEGIES and eos_id is None:
            raise ValueError("LP-MDM evaluation requires validation.sampling.eos_id or training.eos_id.")

        if is_main:
            print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint is missing required key 'model_state_dict': {resume_path}")

        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"], strict=True)
        global_step = ckpt.get("global_step", 0)
        epoch = ckpt.get("epoch", 0)

        if train_cfg.ema is not None:
            if "ema_state_dict" not in ckpt:
                raise KeyError(
                    "training.ema is enabled, but checkpoint is missing "
                    f"required key 'ema_state_dict': {resume_path}"
                )
            assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
            ema_params = [p for p in model_to_load.parameters() if p.requires_grad]
            ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.move_shadow_params_to_device(device)
            ema.copy_to(model_to_load.parameters())
            if is_main:
                print("Loaded EMA weights from checkpoint.")
        elif is_main:
            print("Loaded model weights from checkpoint.")

        if is_main:
            print(f"Checkpoint metadata: epoch {epoch}, global_step {global_step}")
        if cfg.validation.get("eval_itr", None) is None:
            cfg.validation.eval_itr = int(global_step)

        if cfg.wandb.wandb and is_main:
            wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)
            wandb_initialized = True

        model.eval()
        with torch.inference_mode():
            val_acc_dict = evaluate_ddp_dict(model, cfg, device, rank, world_size)

        if is_main:
            log_prefix = "ema_val_acc_" if train_cfg.ema is not None else "val_acc_"
            for key, value in val_acc_dict.items():
                print(f"Validation Accuracy {key}: {value}")
                if cfg.wandb.wandb:
                    wandb.log({log_prefix + key: value}, step=global_step)

        return val_acc_dict
    finally:
        if wandb_initialized:
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

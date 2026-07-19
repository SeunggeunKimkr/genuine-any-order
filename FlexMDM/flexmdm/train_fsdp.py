"""FSDP entrypoint for the FlexMDM training skeleton."""

from __future__ import annotations

import importlib
import math
import os
import random
import sys
from datetime import timedelta
from functools import partial
from typing import Any, Optional

os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError as exc:
    hydra = None
    DictConfig = Any
    OmegaConf = None
    _HYDRA_IMPORT_ERROR = exc
else:
    _HYDRA_IMPORT_ERROR = None


def _print_minimal_help_without_hydra() -> None:
    print(
        "FlexMDM FSDP training skeleton\n\n"
        "Usage:\n"
        "  python -m flexmdm.train_fsdp [HYDRA_OVERRIDES]\n\n"
        "This entrypoint uses Hydra for configuration. The current Python "
        "environment does not have hydra-core installed, so only this minimal "
        "help text is available here."
    )


if hydra is None:
    if any(arg in {"--help", "-h", "--hydra-help"} for arg in sys.argv[1:]):
        _print_minimal_help_without_hydra()
        raise SystemExit(0)
    raise ModuleNotFoundError(
        "hydra-core is required to run FlexMDM FSDP training. "
        "Install the instruct requirements or run from the project training environment."
    ) from _HYDRA_IMPORT_ERROR

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .architecture import build_flexmdm_model
from .trainer import FlexMDMFSDPTrainer


def initialize_distributed() -> tuple[int, int, int]:
    """Initialize torchrun/NCCL state."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlexMDM FSDP training.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    launched_with_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched_with_torchrun and not dist.is_initialized():
        # 60-min collective timeout buys safety margin around long checkpoint
        # writes — even if the rank-0 forked-save subprocess takes longer than
        # the default 10-min watchdog (slow netscratch days), the in-flight
        # all-gathers won't be erroneously aborted. True hangs (a dead node)
        # still get caught, just an hour later.
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=60),
        )
    return local_rank, rank, world_size


def finalize_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: Optional[str]) -> torch.dtype:
    if dtype_name is None:
        return torch.bfloat16
    if not hasattr(torch, str(dtype_name)):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    dtype = getattr(torch, str(dtype_name))
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Not a torch dtype: {dtype_name}")
    return dtype


def build_mixed_precision(config: DictConfig) -> Optional[MixedPrecision]:
    if not bool(config.fsdp.mixed_precision.enabled):
        return None
    return MixedPrecision(
        param_dtype=resolve_torch_dtype(config.fsdp.mixed_precision.param_dtype),
        reduce_dtype=resolve_torch_dtype(config.fsdp.mixed_precision.reduce_dtype),
        buffer_dtype=resolve_torch_dtype(config.fsdp.mixed_precision.buffer_dtype),
        cast_forward_inputs=True,
    )


def build_auto_wrap_policy(config: DictConfig):
    if not bool(config.fsdp.auto_wrap):
        return None
    min_num_params = int(config.fsdp.min_num_params)
    return partial(size_based_auto_wrap_policy, min_num_params=min_num_params)


def build_model_and_tokenizer(config: DictConfig):
    # If resuming, the checkpoint dir is a drop-in HF-sharded source for
    # the backbone (it was saved via `backbone.save_pretrained`), so we point
    # `from_pretrained` at it and additionally load `flexmdm_extras.pt`
    # below. The original `partial_pretrain` is ignored in that case.
    resume_from = config.trainer.get("resume_from", None) if hasattr(
        config, "trainer"
    ) else None
    if resume_from:
        resume_from = os.path.expanduser(str(resume_from))
        model_path = resume_from
    else:
        model_path = os.path.expanduser(str(config.model.partial_pretrain))
    trust_remote_code = bool(config.model.trust_remote_code)

    external_lib = config.model.get("external_lib", None)
    if external_lib:
        importlib.import_module(str(external_lib))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if getattr(tokenizer, "mask_token_id", None) is None:
        raise ValueError(
            f"Tokenizer for {model_path} does not expose mask_token_id; "
            "FlexMDM training requires a registered mask token."
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        attention_dropout=float(config.model.attention_dropout),
    )

    model_kwargs: dict[str, Any] = {
        "config": hf_config,
        "torch_dtype": resolve_torch_dtype(config.model.torch_dtype),
        "trust_remote_code": trust_remote_code,
    }
    attn_implementation = config.model.get("attn_implementation", None)
    if attn_implementation:
        model_kwargs["attn_implementation"] = str(attn_implementation)

    backbone = AutoModel.from_pretrained(model_path, **model_kwargs)
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False

    if bool(config.model.enable_gradient_checkpointing):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    max_length = int(config.data.get("max_length", 768) or 768)
    model = build_flexmdm_model(
        backbone, tokenizer=tokenizer, max_length=max_length
    )
    model.to(dtype=resolve_torch_dtype(config.model.torch_dtype))

    if resume_from:
        extras_path = os.path.join(resume_from, "flexmdm_extras.pt")
        if not os.path.isfile(extras_path):
            raise FileNotFoundError(
                f"resume_from={resume_from!r} does not contain flexmdm_extras.pt"
            )
        extras = torch.load(extras_path, map_location="cpu", weights_only=True)
        incompat = model.load_state_dict(extras, strict=False)
        expected_extras_prefixes = (
            "insertion_head.",
            "time_mlp.",
            "temb_mods.",
        )
        missing_extras = [
            k
            for k in incompat.missing_keys
            if k.startswith(expected_extras_prefixes)
        ]
        if missing_extras:
            raise RuntimeError(
                f"Resume failed: missing FlexMDM extras keys: {missing_extras}"
            )
        if incompat.unexpected_keys:
            raise RuntimeError(
                f"Resume failed: unexpected keys in extras: {incompat.unexpected_keys}"
            )

    return model, tokenizer


def _resolve_sharding(
    config: DictConfig, world_size: int
) -> tuple[ShardingStrategy, Any]:
    """Return (sharding_strategy, device_mesh) for the configured topology.

    For hybrid_shard, builds a 2D device mesh (num_nodes, gpus_per_node) so
    params are only sharded within a node and replicated across nodes, which
    keeps all-gather traffic on fast intra-node NVLink and restricts slow
    inter-node communication to the per-step gradient all-reduce.
    """
    strategy_name = str(config.fsdp.get("sharding_strategy", "full_shard")).lower()
    if strategy_name == "full_shard":
        return ShardingStrategy.FULL_SHARD, None
    if strategy_name == "hybrid_shard":
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        if local_world_size <= 0 or world_size % local_world_size != 0:
            raise ValueError(
                "hybrid_shard requires WORLD_SIZE divisible by LOCAL_WORLD_SIZE, "
                f"got world_size={world_size}, local_world_size={local_world_size}."
            )
        num_nodes = world_size // local_world_size
        # FSDP convention for HYBRID_SHARD: first mesh dim is the replicate
        # (inter-node) dim, second is the shard (intra-node) dim.
        device_mesh = init_device_mesh(
            "cuda",
            (num_nodes, local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )
        return ShardingStrategy.HYBRID_SHARD, device_mesh
    raise ValueError(
        f"Unknown fsdp.sharding_strategy: {strategy_name!r} "
        "(expected 'full_shard' or 'hybrid_shard')"
    )


def build_fsdp_model(
    *,
    model: torch.nn.Module,
    config: DictConfig,
    device: torch.device,
    world_size: int,
) -> FSDP:
    cpu_offload = None
    if bool(config.fsdp.cpu_offload):
        cpu_offload = CPUOffload(
            offload_params=bool(config.fsdp.offload_params)
        )

    sharding_strategy, device_mesh = _resolve_sharding(config, world_size)

    fsdp_kwargs: dict[str, Any] = dict(
        module=model,
        auto_wrap_policy=build_auto_wrap_policy(config),
        sharding_strategy=sharding_strategy,
        mixed_precision=build_mixed_precision(config),
        cpu_offload=cpu_offload,
        device_id=device,
        sync_module_states=world_size > 1
        and bool(config.fsdp.sync_module_states),
        use_orig_params=bool(config.fsdp.use_orig_params),
    )
    if device_mesh is not None:
        fsdp_kwargs["device_mesh"] = device_mesh
    return FSDP(**fsdp_kwargs)


INSERTION_MODULE_FRAGMENTS = ("insertion_head", "time_mlp", "temb_mods")


def _is_insertion_param(name: str) -> bool:
    """Return True if *name* belongs to one of the FlexMDM insertion modules."""
    return any(fragment in name for fragment in INSERTION_MODULE_FRAGMENTS)


def build_optimizer(
    config: DictConfig, fsdp_model: FSDP, *, rank: int = 0
) -> torch.optim.Optimizer:
    base_lr = float(config.optim.lr)
    multiplier = float(config.optim.get("insertion_lr_multiplier", 1.0))
    insertion_lr = base_lr * multiplier

    backbone_params: list[torch.nn.Parameter] = []
    insertion_params: list[torch.nn.Parameter] = []
    for name, param in fsdp_model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_insertion_param(name):
            insertion_params.append(param)
        else:
            backbone_params.append(param)

    if not insertion_params:
        raise RuntimeError(
            "No insertion-module parameters found — FSDP parameter names may "
            "have changed. First 500 chars of names seen: "
            + ", ".join(n for n, _ in fsdp_model.named_parameters())[:500]
        )

    if rank == 0:
        n_backbone = sum(p.numel() for p in backbone_params)
        n_insertion = sum(p.numel() for p in insertion_params)
        print(
            f"Optimizer param groups: backbone={n_backbone:,} params (lr={base_lr}), "
            f"insertion={n_insertion:,} params (lr={insertion_lr})",
            flush=True,
        )

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": base_lr},
            {"params": insertion_params, "lr": insertion_lr},
        ],
        betas=tuple(float(beta) for beta in config.optim.betas),
        weight_decay=float(config.optim.weight_decay),
    )


def cosine_warmup_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(
    config: DictConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(config.trainer.total_training_steps))
    warmup_steps = int(total_steps * float(config.optim.warmup_steps_ratio))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(
            cosine_warmup_lambda,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )


def print_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


@hydra.main(config_path="config", config_name="fsdp_train", version_base=None)
def main(config: DictConfig) -> None:
    local_rank, rank, world_size = initialize_distributed()
    device = torch.device("cuda", local_rank)
    set_seed(int(config.trainer.seed), rank)

    try:
        print_rank0(rank, OmegaConf.to_yaml(config))
        model, tokenizer = build_model_and_tokenizer(config)
        fsdp_model = build_fsdp_model(
            model=model,
            config=config,
            device=device,
            world_size=world_size,
        )
        optimizer = build_optimizer(config, fsdp_model, rank=rank)
        lr_scheduler = build_scheduler(config, optimizer)

        trainer = FlexMDMFSDPTrainer(
            config=config,
            fsdp_model=fsdp_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        trainer.fit()
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()

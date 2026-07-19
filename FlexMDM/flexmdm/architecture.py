"""Architecture wrapper for turning Dream-Coder into FlexMDM.

This module intentionally contains only model architecture additions:
time conditioning and the insertion head. Losses, interpolants, schedules,
inference, and data preparation belong in later FlexMDM modules.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn


class InsertionHead(nn.Module):
    """Predict the log of the per-position insertion rate.

    Returns `z ∈ ℝ`, clamped to [-15, 15] to keep `exp(z)` safely
    representable in bf16. The actual rate is `exp(z)`, computed by
    downstream consumers.
    """

    Z_CLAMP_MIN = -15.0
    Z_CLAMP_MAX = 15.0

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.proj2.weight, std=1e-3)
        nn.init.constant_(self.proj2.bias, -1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.act(self.proj1(x))
        x = self.proj2(x).squeeze(-1)
        return x.clamp(min=self.Z_CLAMP_MIN, max=self.Z_CLAMP_MAX)


class FlexMDM(nn.Module):
    """Dream backbone wrapper with timestep AdaLN conditioning and insertion head."""

    def __init__(
        self,
        backbone: nn.Module,
        pad_token_id: Optional[int] = None,
        hidden_size: Optional[int] = None,
        max_length: int = 768,
        time_scale: float = 1000.0,
        time_max_period: int = 10000,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = self._resolve_hidden_size(backbone, hidden_size)
        self.pad_token_id = pad_token_id
        self.max_length = int(max_length)
        self.time_scale = float(time_scale)
        self.time_max_period = int(time_max_period)
        self.insertion_head = InsertionHead(self.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self._temb: Optional[torch.Tensor] = None
        self._hook_handles: list[Any] = []
        self._wrap_adaln()

    @staticmethod
    def _resolve_hidden_size(
        backbone: nn.Module, hidden_size: Optional[int]
    ) -> int:
        if hidden_size is not None:
            return int(hidden_size)
        config = getattr(backbone, "config", None)
        if config is not None and getattr(config, "hidden_size", None) is not None:
            return int(config.hidden_size)
        raise ValueError(
            "FlexMDM requires hidden_size or backbone.config.hidden_size."
        )

    @staticmethod
    def _decoder_layers(backbone: nn.Module) -> nn.ModuleList:
        base_model = getattr(backbone, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None:
            raise ValueError(
                "FlexMDM expects a Dream-style backbone with backbone.model.layers."
            )
        return layers

    def _wrap_adaln(self) -> None:
        self.temb_mods = nn.ModuleList()
        for layer in self._decoder_layers(self.backbone):
            mod = nn.Linear(self.hidden_size, self.hidden_size * 2)
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
            self.temb_mods.append(mod)

            for norm_name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, norm_name, None)
                if norm is None:
                    raise ValueError(
                        "FlexMDM expects each Dream layer to expose "
                        f"{norm_name}."
                    )
                handle = norm.register_forward_hook(self._adaln_hook(mod))
                self._hook_handles.append(handle)

    @staticmethod
    def _module_weight_device_dtype(
        module: nn.Module,
        *,
        default_device: torch.device,
        default_dtype: torch.dtype,
    ) -> tuple[torch.device, torch.dtype]:
        for submodule in module.modules():
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight.device, weight.dtype
        for parameter in module.parameters():
            return parameter.device, parameter.dtype
        return default_device, default_dtype

    def _adaln_hook(self, mod: nn.Module):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], out: torch.Tensor):
            if self._temb is None:
                return out
            mod_device, mod_dtype = self._module_weight_device_dtype(
                mod,
                default_device=out.device,
                default_dtype=out.dtype,
            )
            temb = self._temb.to(device=mod_device, dtype=mod_dtype)
            scale, shift = mod(temb).chunk(2, dim=-1)
            scale = scale.to(device=out.device, dtype=out.dtype)
            shift = shift.to(device=out.device, dtype=out.dtype)
            return (1 + scale[:, None, :]) * out + shift[:, None, :]

        return hook

    def remove_adaln_hooks(self) -> None:
        """Remove registered AdaLN hooks, mainly for tests or teardown."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def clear_conditioning_cache(self) -> None:
        """Clear cached timestep conditioning after backward/recompute is done."""
        self._temb = None

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape (batch,), got {tuple(timesteps.shape)}."
            )

        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(self.time_max_period)
            * torch.arange(
                start=0,
                end=half,
                dtype=torch.float32,
                device=timesteps.device,
            )
            / half
        )
        args = (timesteps.float() * self.time_scale)[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.hidden_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(
        self,
        input_ids: torch.LongTensor,
        timesteps: torch.Tensor,
        **backbone_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided for FlexMDM.")
        if timesteps is None:
            raise ValueError("timesteps must be provided for FlexMDM.")
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape (batch,), got {tuple(timesteps.shape)}."
            )
        if timesteps.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "timesteps batch size must match input_ids batch size: "
                f"{timesteps.shape[0]} != {input_ids.shape[0]}."
            )

        temb = self.timestep_embedding(timesteps.to(input_ids.device))
        time_mlp_device, time_mlp_dtype = self._module_weight_device_dtype(
            self.time_mlp,
            default_device=input_ids.device,
            default_dtype=temb.dtype,
        )
        temb = temb.to(device=time_mlp_device, dtype=time_mlp_dtype)
        self._temb = self.time_mlp(temb)

        backbone_kwargs.setdefault("return_dict", True)
        # Dream-Coder's attention path forwards attention_mask directly to SDPA,
        # which needs a 4D mask broadcastable to [B, H, S_q, S_k]. A 2D [B, S]
        # padding mask only broadcasts when B == 1; expand to [B, 1, 1, S] so
        # it works for any batch size.
        attention_mask = backbone_kwargs.get("attention_mask", None)
        if attention_mask is not None and attention_mask.dim() == 2:
            backbone_kwargs["attention_mask"] = attention_mask.bool()[:, None, None, :]
        outputs = self.backbone.model(input_ids=input_ids, **backbone_kwargs)
        last_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        logits = self.backbone.lm_head(last_hidden)
        log_length = self.insertion_head(last_hidden)
        return {"logits": logits, "log_length": log_length}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def _resolve_pad_token_id(
    backbone: nn.Module,
    tokenizer: Optional[Any] = None,
    pad_token_id: Optional[int] = None,
) -> Optional[int]:
    if pad_token_id is not None:
        return int(pad_token_id)
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id)
    config = getattr(backbone, "config", None)
    if config is not None and getattr(config, "pad_token_id", None) is not None:
        return int(config.pad_token_id)
    return None


def build_flexmdm_model(
    backbone: nn.Module,
    tokenizer: Optional[Any] = None,
    pad_token_id: Optional[int] = None,
    max_length: int = 768,
) -> FlexMDM:
    """Wrap a Dream-Coder backbone with FlexMDM architecture additions."""
    return FlexMDM(
        backbone=backbone,
        pad_token_id=_resolve_pad_token_id(backbone, tokenizer, pad_token_id),
        max_length=max_length,
    )


__all__ = ["FlexMDM", "InsertionHead", "build_flexmdm_model"]

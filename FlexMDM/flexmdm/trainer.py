"""FlexMDM trainer and loss helpers."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .data import (
    build_train_dataloader,
    build_validation_dataloader,
)
from .schedules import (
    linear_elbo_weight,
    sample_globally_stratified_time,
    sample_linear_hitting_times,
    sample_linear_time,
    sample_schedule_times,
    schedule_elbo_weights,
)


def poisson_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Poisson NLL up to a prediction-independent constant.

    x: target counts (non-negative). z: predicted log-rate. Both broadcast
    element-wise. Returns exp(z) - x * z, computed in float32 for stable
    accumulation (bf16 exp can round coarsely near the tails).
    """
    z32 = z.float()
    return z32.exp() - x.float() * z32


class FlexMDMProcess:
    def __init__(
        self,
        vocab_size: int,
        mask_id: int,
        pad_id: int,
        max_len: int,
        eps: float = 1e-6,
        insertion_schedule: str = "linear",
        unmasking_schedule: str = "linear",
        insertion_exponent: Optional[float] = None,
        unmasking_exponent: Optional[float] = None,
        insertion_params: Optional[Mapping[str, Any]] = None,
        unmasking_params: Optional[Mapping[str, Any]] = None,
    ):
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.scale_factor = max_len
        self.eps = float(eps)
        self.insertion_schedule = insertion_schedule
        self.unmasking_schedule = unmasking_schedule
        self.insertion_exponent = (
            float(insertion_exponent) if insertion_exponent is not None else None
        )
        self.unmasking_exponent = (
            float(unmasking_exponent) if unmasking_exponent is not None else None
        )
        self.insertion_params = (
            dict(insertion_params) if insertion_params is not None else None
        )
        self.unmasking_params = (
            dict(unmasking_params) if unmasking_params is not None else None
        )

    def mask_indices(self, xt: torch.Tensor) -> torch.Tensor:
        return xt == self.mask_id

    def length(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.bool().sum(dim=1)

    def gathered_unmasked(self, x1: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        st_safe = st.clamp(min=0)
        return torch.gather(x1, 1, st_safe)

    def gaps_and_masks(
        self,
        x1: torch.Tensor,
        st: torch.Tensor,
        prompt_mask: torch.BoolTensor,
        x1_attention_mask: torch.BoolTensor,
        xt_attention_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return after-token insertion targets and mask.

        The conceptual gaps are:
        before token 0, after token 0, ..., after token L - 1.
        We drop the first gap and keep the after-token gaps, so the returned
        tensors align with the model's B x L insertion head.
        """
        x1_attention_mask = x1_attention_mask.bool()
        xt_attention_mask = xt_attention_mask.bool()
        prompt_mask = prompt_mask.bool()
        x1_len = self.length(x1_attention_mask)
        xt_len = self.length(xt_attention_mask)

        original_idx = st.clamp(min=0)
        next_original_idx = torch.cat(
            [original_idx[:, 1:], x1_len.unsqueeze(1)],
            dim=1,
        )
        last_visible = (xt_len - 1).clamp_min(0).unsqueeze(1)
        next_original_idx.scatter_(1, last_visible, x1_len.unsqueeze(1))

        gaps = next_original_idx - original_idx - 1
        gaps = gaps.clamp(min=0)
        gaps = gaps.masked_fill(~xt_attention_mask, 0)

        prompt_len = self.length(prompt_mask)
        first_valid_original_idx = (prompt_len - 1).clamp_min(0)
        has_answer = (x1_attention_mask & ~prompt_mask).any(dim=1)
        insertion_mask = (
            xt_attention_mask
            & has_answer.unsqueeze(1)
            & (original_idx >= first_valid_original_idx.unsqueeze(1))
        )

        return gaps, insertion_mask

    def sample_times(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_schedule_times(
            x1,
            insertion_schedule=self.insertion_schedule,
            unmasking_schedule=self.unmasking_schedule,
            eps=self.eps,
            insertion_exponent=self.insertion_exponent,
            unmasking_exponent=self.unmasking_exponent,
            insertion_params=self.insertion_params,
            unmasking_params=self.unmasking_params,
        )

    def flexmdm_process(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        mask: torch.BoolTensor,
        attention_mask: torch.BoolTensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if x1.ndim != 2:
            raise ValueError(f"x1 must have shape (B, L), got {tuple(x1.shape)}")
        if x1.shape[1] > self.max_len:
            raise ValueError(
                f"x1 length {x1.shape[1]} exceeds configured max_len {self.max_len}"
            )
        if mask.shape != x1.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match x1 shape {tuple(x1.shape)}"
            )
        if attention_mask.shape != x1.shape:
            raise ValueError(
                "attention_mask shape "
                f"{tuple(attention_mask.shape)} must match x1 shape {tuple(x1.shape)}"
            )
        if t.shape != (x1.shape[0],):
            raise ValueError(
                f"t must have shape ({x1.shape[0]},), got {tuple(t.shape)}"
            )

        mask = mask.bool()
        attention_mask = attention_mask.bool()
        if (mask & ~attention_mask).any():
            raise ValueError("prompt/fixed mask must be a subset of attention_mask")

        insertion_time, unmasking_time = self.sample_times(x1)
        valid_tokens = attention_mask
        deleted_tokens = valid_tokens & (t[:, None] < insertion_time) & (~mask)
        masked_tokens = (
            valid_tokens
            & (t[:, None] >= insertion_time)
            & (t[:, None] < unmasking_time)
            & (~mask)
        )
        xt_attention_mask = attention_mask & ~deleted_tokens
        xt = torch.where(
            deleted_tokens,
            self.pad_id,
            torch.where(masked_tokens, self.mask_id, x1),
        )
        # index set st: real tokens first (stable), then inactive padding.
        st = (
            xt_attention_mask
            .to(torch.int32)
            .argsort(dim=1, descending=True, stable=True)
        )
        xt = torch.gather(xt, 1, st)
        xt_attention_mask = torch.gather(xt_attention_mask, 1, st)
        st[~xt_attention_mask] = -1

        masked_indices = self.mask_indices(xt) & xt_attention_mask
        gaps, insertion_mask = self.gaps_and_masks(
            x1,
            st,
            mask,
            attention_mask,
            xt_attention_mask,
        )

        return xt, xt_attention_mask, st, x1, masked_indices, gaps, insertion_mask

    def elbo_weights(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cap weights at roughly 1/1e-3 to prevent loss spikes when stratified
        # t sampling lands very close to 1.
        return schedule_elbo_weights(
            t,
            insertion_schedule=self.insertion_schedule,
            unmasking_schedule=self.unmasking_schedule,
            eps=max(self.eps, 1e-3),
            insertion_exponent=self.insertion_exponent,
            unmasking_exponent=self.unmasking_exponent,
            insertion_params=self.insertion_params,
            unmasking_params=self.unmasking_params,
        )

    def loss(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        mask: torch.BoolTensor,
        attention_mask: torch.BoolTensor,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FlexMDM losses.

        Supports models that return either:
        - Tensor logits: [B, L, vocab_size]
        - Dict with keys: "logits" and optional "length"
        """
        if t is None:
            t = sample_linear_time(
                x1.shape[0],
                device=x1.device,
                dtype=torch.float32,
                eps=self.eps,
            )
        else:
            t = t.to(device=x1.device, dtype=torch.float32)

        (
            xt,
            xt_attention_mask,
            st,
            x1,
            masked_indices,
            gaps,
            insertion_mask,
        ) = self.flexmdm_process(
            x1,
            t,
            mask,
            attention_mask,
        )

        out = model(xt, t, attention_mask=xt_attention_mask)
        scale_factor = x1.shape[0] * self.scale_factor

        if isinstance(out, dict):
            logits = out["logits"]
            log_length_pred = out.get("log_length", None)
        else:
            logits = out
            log_length_pred = None

        insertion_weight_t, unmasking_weight_t = self.elbo_weights(t)
        unmasking_weight = unmasking_weight_t[:, None].expand_as(
            masked_indices
        )
        insertion_weight = insertion_weight_t[:, None].expand_as(
            insertion_mask
        )

        unmask_pred = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        target_tokens = self.gathered_unmasked(x1, st)
        unmask_loss = (
            unmasking_weight[masked_indices]
            * F.cross_entropy(
                unmask_pred[masked_indices],
                target_tokens[masked_indices],
                reduction="none",
            )
        ).sum() / scale_factor

        if log_length_pred is not None:
            insertion_loss = (
                insertion_weight[insertion_mask].float()
                * poisson_loss(
                    gaps[insertion_mask],
                    log_length_pred[insertion_mask],
                )
            ).sum() / scale_factor
        else:
            insertion_loss = logits.new_zeros(())

        return unmask_loss, insertion_loss


class FlexMDMFSDPTrainer:
    """FSDP trainer for full-model FlexMDM fine-tuning."""

    def __init__(
        self,
        *,
        config: Any,
        fsdp_model: FSDP,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        rank: int,
        world_size: int,
    ):
        self.config = config
        self.fsdp_model = fsdp_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.global_step = 0
        self.loss_process: Optional[FlexMDMProcess] = None
        self.last_loss_metrics: Dict[str, float] = {}
        self._wandb_run = None
        # Updated by the training loop; consumed by save_checkpoint so a
        # resume can pick up at the same dataloader position.
        self._current_epoch = 0
        self._next_batch_in_epoch = 0
        # Set by load_checkpoint and consumed once by fit().
        self._resume_epoch = 0
        self._resume_batch_index = 0
        # Forked subprocess pid for rank-0 disk writes during save_checkpoint.
        # We fork rather than thread because torch.save's pickle path holds
        # the Python GIL for the duration of large-tensor writes, which would
        # starve the main thread of CPU time and block its NCCL kernel
        # launches. A subprocess has its own interpreter/GIL so its disk
        # work cannot stall the parent's training collectives.
        self._pending_save_pid: Optional[int] = None
        # Co-tracked with _pending_save_pid so the next _await call can
        # name the partial dir/step when it logs a failure.
        self._pending_save_dir: Optional[str] = None
        self._pending_save_step: Optional[int] = None

    def compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute total FlexMDM loss for a batch."""
        input_ids = self._batch_get(batch, "input_ids")
        if input_ids is None:
            raise KeyError("FlexMDM batches must include input_ids")
        input_ids = input_ids.to(self.device, non_blocking=True).long()

        prompt_mask = self._resolve_prompt_mask(batch).to(
            self.device, non_blocking=True
        )
        prompt_mask = prompt_mask.bool()

        attention_mask = self._resolve_attention_mask(batch).to(
            self.device, non_blocking=True
        )
        attention_mask = attention_mask.bool()

        t = self._batch_get(batch, "t")
        if t is None:
            t = self._batch_get(batch, "timesteps")
        if t is not None:
            t = t.to(self.device, non_blocking=True)

        process = self._get_loss_process(seq_len=input_ids.shape[1])
        unmask_loss, insertion_loss = process.loss(
            self.fsdp_model,
            input_ids,
            prompt_mask,
            attention_mask,
            t=t,
        )
        total_loss = unmask_loss + insertion_loss
        self.last_loss_metrics = {
            "train/unmask_loss": float(unmask_loss.detach().item()),
            "train/insertion_loss": float(insertion_loss.detach().item()),
            "train/loss": float(total_loss.detach().item()),
        }
        return total_loss

    @staticmethod
    def _batch_get(batch: Any, key: str) -> Optional[torch.Tensor]:
        if isinstance(batch, dict):
            return batch.get(key)
        try:
            if key in batch:
                return batch[key]
        except Exception:
            pass
        return getattr(batch, key, None)

    @staticmethod
    def _slice_batch(batch: Any, start: int, stop: int) -> Any:
        if isinstance(batch, dict):
            sliced: Dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    sliced[key] = value[start:stop]
                else:
                    sliced[key] = value
            return sliced
        return batch[start:stop]

    def _resolve_prompt_mask(self, batch: Any) -> torch.Tensor:
        prompt_mask = self._batch_get(batch, "mask")
        if prompt_mask is not None:
            return prompt_mask
        prompt_mask = self._batch_get(batch, "prompt_mask")
        if prompt_mask is not None:
            return prompt_mask
        loss_mask = self._batch_get(batch, "loss_mask")
        if loss_mask is not None:
            return ~loss_mask.bool()
        raise KeyError(
            "FlexMDM batches must include mask, prompt_mask, or loss_mask"
        )

    def _resolve_attention_mask(self, batch: Any) -> torch.Tensor:
        attention_mask = self._batch_get(batch, "attention_mask")
        if attention_mask is not None:
            return attention_mask
        attention_mask = self._batch_get(batch, "attn_mask")
        if attention_mask is not None:
            return attention_mask
        raise KeyError("FlexMDM batches must include attention_mask")

    def _get_loss_process(self, seq_len: int) -> FlexMDMProcess:
        max_len = int(self._config_get("data", "max_length", default=seq_len))
        if self.loss_process is not None and self.loss_process.max_len == max_len:
            return self.loss_process

        insertion_schedule = self._flexmdm_schedule_choice("insertion_schedule")
        unmasking_schedule = self._flexmdm_schedule_choice("unmasking_schedule")
        insertion_exponent, unmasking_exponent = self._flexmdm_schedule_exponents(
            insertion_schedule, unmasking_schedule
        )
        insertion_params, unmasking_params = self._flexmdm_schedule_params(
            insertion_schedule, unmasking_schedule
        )
        self.loss_process = FlexMDMProcess(
            vocab_size=self._resolve_vocab_size(),
            mask_id=self._resolve_mask_token_id(),
            pad_id=self._resolve_pad_token_id(),
            max_len=max_len,
            insertion_schedule=insertion_schedule,
            unmasking_schedule=unmasking_schedule,
            insertion_exponent=insertion_exponent,
            unmasking_exponent=unmasking_exponent,
            insertion_params=insertion_params,
            unmasking_params=unmasking_params,
        )
        return self.loss_process

    def _flexmdm_schedule_choice(self, schedule_key: str) -> str:
        schedule = self._config_get("flexmdm", schedule_key, default=None)
        if schedule is None:
            return "linear"
        if isinstance(schedule, str):
            return schedule
        if isinstance(schedule, dict):
            value = schedule.get("schedule", None)
            if value is None:
                value = schedule.get("sampling", None)
        else:
            value = getattr(schedule, "schedule", None)
            if value is None:
                value = getattr(schedule, "sampling", None)
        if value is None:
            return "linear"
        return str(value)

    def _flexmdm_schedule_exponents(
        self,
        insertion_schedule: str,
        unmasking_schedule: str,
    ) -> tuple[Optional[float], Optional[float]]:
        """Resolve (insertion_exponent, unmasking_exponent) from config.

        Priority: ``flexmdm.insertion_exponent`` / ``flexmdm.unmasking_exponent``
        override everything. Otherwise, if either schedule is ``power``, read
        ``flexmdm.schedule_a`` and ``flexmdm.schedule_b`` (defaults 1.8) and
        compute insertion=a, unmasking=a*b per the family
        alpha_t=1-(1-t)^a, beta_t=1-(1-t)^(a*b).
        """
        explicit_ins = self._config_get("flexmdm", "insertion_exponent", default=None)
        explicit_unm = self._config_get("flexmdm", "unmasking_exponent", default=None)
        needs_a_b = (
            insertion_schedule == "power" and explicit_ins is None
        ) or (
            unmasking_schedule == "power" and explicit_unm is None
        )
        a = b = None
        if needs_a_b:
            a = float(self._config_get("flexmdm", "schedule_a", default=1.8))
            b = float(self._config_get("flexmdm", "schedule_b", default=1.8))

        def _pick(schedule: str, explicit: Any, fallback: Optional[float]) -> Optional[float]:
            if explicit is not None:
                return float(explicit)
            if schedule == "power":
                return fallback
            return None

        return (
            _pick(insertion_schedule, explicit_ins, a),
            _pick(unmasking_schedule, explicit_unm, (a * b) if (a is not None and b is not None) else None),
        )

    def _flexmdm_schedule_params(
        self,
        insertion_schedule: str,
        unmasking_schedule: str,
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Resolve (insertion_params, unmasking_params) from config.

        Reads shared shortcuts and per-side overrides:
          - log_linear: ``flexmdm.log_linear_lambda`` / ``flexmdm.log_linear_c``
            with per-side ``flexmdm.insertion_log_linear_*`` /
            ``flexmdm.unmasking_log_linear_*`` overrides.
          - logit_power: ``flexmdm.logit_power_a`` / ``flexmdm.logit_power_b``
            with per-side ``flexmdm.insertion_logit_power_*`` /
            ``flexmdm.unmasking_logit_power_*`` overrides.

        Returns ``None`` for any side that uses a power-family schedule, so
        the schedules library falls back to its defaults if the user picks
        the new schedule without setting any keys.
        """

        def _pick(side: str, schedule: str) -> Optional[Dict[str, float]]:
            if schedule == "log_linear":
                shared_lam = self._config_get("flexmdm", "log_linear_lambda", default=None)
                shared_c = self._config_get("flexmdm", "log_linear_c", default=None)
                side_lam = self._config_get(
                    "flexmdm", f"{side}_log_linear_lambda", default=None
                )
                side_c = self._config_get(
                    "flexmdm", f"{side}_log_linear_c", default=None
                )
                resolved = {}
                lam = side_lam if side_lam is not None else shared_lam
                c = side_c if side_c is not None else shared_c
                if lam is not None:
                    resolved["lam"] = float(lam)
                if c is not None:
                    resolved["c"] = float(c)
                return resolved or None
            if schedule == "logit_power":
                shared_a = self._config_get("flexmdm", "logit_power_a", default=None)
                shared_b = self._config_get("flexmdm", "logit_power_b", default=None)
                side_a = self._config_get(
                    "flexmdm", f"{side}_logit_power_a", default=None
                )
                side_b = self._config_get(
                    "flexmdm", f"{side}_logit_power_b", default=None
                )
                resolved = {}
                a_val = side_a if side_a is not None else shared_a
                b_val = side_b if side_b is not None else shared_b
                if a_val is not None:
                    resolved["a"] = float(a_val)
                if b_val is not None:
                    resolved["b"] = float(b_val)
                return resolved or None
            return None

        return (
            _pick("insertion", insertion_schedule),
            _pick("unmasking", unmasking_schedule),
        )

    def _config_get(self, *path: str, default: Any = None) -> Any:
        value: Any = self.config
        for key in path:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                value = getattr(value, key, default)
            if value is default:
                return default
        return value

    def _resolve_vocab_size(self) -> int:
        if getattr(self.tokenizer, "vocab_size", None) is not None:
            return int(self.tokenizer.vocab_size)
        try:
            return int(len(self.tokenizer))
        except TypeError:
            pass

        module = getattr(self.fsdp_model, "module", self.fsdp_model)
        config = getattr(module, "config", None)
        if config is None and hasattr(module, "backbone"):
            config = getattr(module.backbone, "config", None)
        if config is not None and getattr(config, "vocab_size", None) is not None:
            return int(config.vocab_size)
        raise ValueError("Could not resolve vocab size for FlexMDM loss")

    def _resolve_mask_token_id(self) -> int:
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise ValueError(
                "FlexMDM loss requires tokenizer.mask_token_id to be set."
            )
        return int(mask_id)

    def _resolve_pad_token_id(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self._config_get("data", "pad_id", default=None)
        if pad_id is None:
            pad_id = self._config_get("data", "pad_token_id", default=None)
        if pad_id is None:
            raise ValueError(
                "FlexMDM loss requires tokenizer.pad_token_id or data.pad_id"
            )
        return int(pad_id)

    def _all_reduce_mean(self, values: list[float]) -> list[float]:
        """Average a fixed-length list of scalars across all ranks.

        A single tensor is reduced to keep communication to one collective.
        Falls back to the identity when distributed is not initialized or
        world size is 1.
        """
        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return list(values)
        tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(self.world_size)
        return tensor.tolist()

    def _clear_model_conditioning_cache(self) -> None:
        module = getattr(self.fsdp_model, "module", self.fsdp_model)
        clear_cache = getattr(module, "clear_conditioning_cache", None)
        if clear_cache is not None:
            clear_cache()

    def training_step(self, batch: Any) -> Dict[str, float]:
        """Run one training step with micro-batch gradient accumulation."""
        self.fsdp_model.train()
        self.optimizer.zero_grad(set_to_none=True)

        input_ids = self._batch_get(batch, "input_ids")
        if input_ids is None:
            raise KeyError("FlexMDM batches must include input_ids")
        batch_size = int(input_ids.shape[0])

        base_seed = int(self._config_get("trainer", "seed", default=0) or 0)
        t_all = sample_globally_stratified_time(
            per_rank_batch_size=batch_size,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            step=self.global_step,
            base_seed=base_seed,
        )
        if isinstance(batch, dict):
            batch = {**batch, "t": t_all}
        else:
            try:
                batch["t"] = t_all
            except Exception:
                setattr(batch, "t", t_all)

        micro_bs = int(
            self._config_get(
                "data", "micro_batch_size_per_gpu", default=batch_size
            )
            or batch_size
        )
        if micro_bs <= 0:
            micro_bs = batch_size
        num_micro = max(1, (batch_size + micro_bs - 1) // micro_bs)

        step_loss = 0.0
        step_unmask = 0.0
        step_insertion = 0.0
        for micro_idx in range(num_micro):
            start = micro_idx * micro_bs
            stop = min(start + micro_bs, batch_size)
            if start >= stop:
                continue
            micro_batch = self._slice_batch(batch, start, stop)
            try:
                loss = self.compute_loss(micro_batch) / num_micro
                loss.backward()
            finally:
                self._clear_model_conditioning_cache()
            step_loss += float(loss.detach().item())
            step_unmask += (
                self.last_loss_metrics.get("train/unmask_loss", 0.0) / num_micro
            )
            step_insertion += (
                self.last_loss_metrics.get("train/insertion_loss", 0.0)
                / num_micro
            )

        clip_grad = float(self._config_get("optim", "clip_grad", default=0.0))
        if clip_grad > 0.0:
            grad_norm_tensor = self.fsdp_model.clip_grad_norm_(clip_grad)
            grad_norm = float(grad_norm_tensor)
        else:
            grad_norm = 0.0

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.lr_scheduler is not None:
            last_lrs = self.lr_scheduler.get_last_lr()
            current_lr = float(last_lrs[0])
            insertion_lr = float(last_lrs[1]) if len(last_lrs) > 1 else current_lr
        else:
            current_lr = float(self._config_get("optim", "lr", default=0.0))
            insertion_lr = current_lr

        step_loss, step_unmask, step_insertion = self._all_reduce_mean(
            [step_loss, step_unmask, step_insertion]
        )

        self.last_loss_metrics = {
            "train/loss": step_loss,
            "train/unmask_loss": step_unmask,
            "train/insertion_loss": step_insertion,
            "train/grad_norm": grad_norm,
            "train/lr": current_lr,
            "train/insertion_lr": insertion_lr,
        }
        return dict(self.last_loss_metrics)

    def validation_step(self, batch: Any) -> Dict[str, float]:
        """Run one validation step with stratified t sampling.

        For each sample, ``num_strata`` time values are drawn — one from each
        of ``num_strata`` equal-width sub-intervals of [0, 1). The per-sample
        losses from the ``num_strata`` forward passes are averaged to form a
        lower-variance ELBO estimate. The per-rank validation batch is
        processed in micro-batches of ``data.validation_micro_batch_size_per_gpu``
        (falling back to the training micro-batch size) to keep GPU memory
        bounded while still allowing a larger outer ``validation_batch_size``.
        """
        self.fsdp_model.eval()

        input_ids = self._batch_get(batch, "input_ids")
        if input_ids is None:
            raise KeyError("FlexMDM batches must include input_ids")
        batch_size = int(input_ids.shape[0])

        num_strata = int(
            self._config_get("data", "validation_num_t_strata", default=64) or 64
        )
        if num_strata < 1:
            num_strata = 1

        train_micro_bs = int(
            self._config_get(
                "data", "micro_batch_size_per_gpu", default=batch_size
            )
            or batch_size
        )
        micro_bs = int(
            self._config_get(
                "data",
                "validation_micro_batch_size_per_gpu",
                default=train_micro_bs,
            )
            or train_micro_bs
        )
        if micro_bs <= 0:
            micro_bs = batch_size
        num_micro = max(1, (batch_size + micro_bs - 1) // micro_bs)

        sum_loss = 0.0
        sum_unmask = 0.0
        sum_insertion = 0.0
        with torch.no_grad():
            for k in range(num_strata):
                noise = torch.rand(
                    batch_size, device=self.device, dtype=torch.float32
                )
                t_k = (float(k) + noise) / float(num_strata)
                t_k = t_k.clamp(max=1.0 - 1e-6)

                if isinstance(batch, dict):
                    batch_with_t = {**batch, "t": t_k}
                else:
                    batch_with_t = batch
                    try:
                        batch_with_t["t"] = t_k
                    except Exception:
                        setattr(batch_with_t, "t", t_k)

                k_loss = 0.0
                k_unmask = 0.0
                k_insertion = 0.0
                for micro_idx in range(num_micro):
                    start = micro_idx * micro_bs
                    stop = min(start + micro_bs, batch_size)
                    if start >= stop:
                        continue
                    micro = self._slice_batch(batch_with_t, start, stop)
                    try:
                        loss = self.compute_loss(micro)
                    finally:
                        self._clear_model_conditioning_cache()
                    # compute_loss returns per-token mean over its micro; to
                    # form the per-token mean over the full batch, weight by
                    # the micro's sample count.
                    weight = float(stop - start) / float(batch_size)
                    k_loss += float(loss.detach().item()) * weight
                    k_unmask += (
                        float(
                            self.last_loss_metrics.get("train/unmask_loss", 0.0)
                        )
                        * weight
                    )
                    k_insertion += (
                        float(
                            self.last_loss_metrics.get(
                                "train/insertion_loss", 0.0
                            )
                        )
                        * weight
                    )
                    del loss

                sum_loss += k_loss
                sum_unmask += k_unmask
                sum_insertion += k_insertion

        mean_loss = sum_loss / float(num_strata)
        mean_unmask = sum_unmask / float(num_strata)
        mean_insertion = sum_insertion / float(num_strata)

        mean_loss, mean_unmask, mean_insertion = self._all_reduce_mean(
            [mean_loss, mean_unmask, mean_insertion]
        )

        self.last_loss_metrics = {
            "val/loss": mean_loss,
            "val/unmask_loss": mean_unmask,
            "val/insertion_loss": mean_insertion,
        }
        return dict(self.last_loss_metrics)

    def fit(self) -> None:
        """Run the training loop (resume, epochs, validation, checkpoints)."""
        self._init_tracking()
        try:
            resume_from = self._config_get(
                "trainer", "resume_from", default=None
            )
            if resume_from:
                self.load_checkpoint(str(resume_from))

            train_dataloader = build_train_dataloader(
                config=self.config,
                tokenizer=self.tokenizer,
                rank=self.rank,
                world_size=self.world_size,
            )
            validation_dataloader = build_validation_dataloader(
                config=self.config,
                tokenizer=self.tokenizer,
                rank=self.rank,
                world_size=self.world_size,
            )

            total_steps = int(self.config.trainer.total_training_steps)
            save_steps = int(self.config.trainer.save_checkpoint_steps)
            validation_steps = int(
                self._config_get(
                    "trainer",
                    "validation_steps",
                    default=save_steps,
                )
                or save_steps
            )
            start_epoch = self._resume_epoch
            skip_batches = self._resume_batch_index
            # Consume the resume hints — only the first epoch we enter is
            # subject to the in-epoch skip.
            self._resume_epoch = 0
            self._resume_batch_index = 0
            try:
                steps_per_epoch = max(1, len(train_dataloader))
            except TypeError:
                # Streaming / iterable-style loaders don't define __len__.
                steps_per_epoch = 0
            for epoch in range(
                start_epoch, int(self.config.trainer.total_epochs)
            ):
                if hasattr(train_dataloader, "sampler") and hasattr(
                    train_dataloader.sampler, "set_epoch"
                ):
                    train_dataloader.sampler.set_epoch(epoch)
                self._current_epoch = epoch

                epoch_skip = skip_batches if epoch == start_epoch else 0
                for batch_index, batch in enumerate(train_dataloader):
                    if batch_index < epoch_skip:
                        continue
                    self._next_batch_in_epoch = batch_index + 1
                    metrics = self.training_step(batch)
                    self.global_step += 1
                    if steps_per_epoch > 0:
                        metrics["train/epoch_progress"] = (
                            float(epoch) + float(batch_index + 1) / steps_per_epoch
                        )
                    self._log_metrics(metrics, step=self.global_step)

                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(self.global_step)

                    if (
                        validation_dataloader is not None
                        and validation_steps > 0
                        and self.global_step % validation_steps == 0
                        and self.global_step < total_steps
                    ):
                        validation_metrics = self._run_validation(
                            validation_dataloader
                        )
                        if validation_metrics:
                            self._log_metrics(
                                validation_metrics,
                                step=self.global_step,
                            )

                    if self.global_step >= total_steps:
                        validation_metrics = self._run_validation(
                            validation_dataloader
                        )
                        if validation_metrics:
                            self._log_metrics(
                                validation_metrics,
                                step=self.global_step,
                            )
                        self.save_checkpoint(self.global_step)
                        return
        finally:
            # Flush any in-flight forked save before we exit, so the final
            # save is durable on disk. _await_pending_save no longer raises
            # in the normal failure path (logs + marker file instead) — the
            # try/except is kept as defense in depth in case future edits
            # reintroduce a raising path.
            pending_save_exc: Optional[BaseException] = None
            if self._pending_save_pid is not None:
                try:
                    self._await_pending_save()
                except BaseException as exc:
                    pending_save_exc = exc
            self._finish_tracking()
            if pending_save_exc is not None:
                raise pending_save_exc

    def _run_validation(self, validation_dataloader: Any) -> Dict[str, float]:
        if validation_dataloader is None:
            return {}
        sums: Dict[str, float] = {}
        count = 0
        for batch in validation_dataloader:
            metrics = self.validation_step(batch)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            count += 1
        if count == 0:
            return {}
        return {key: value / count for key, value in sums.items()}

    def _logger_names(self) -> list[str]:
        logger = self._config_get("trainer", "logger", default=["console"])
        if isinstance(logger, str):
            return [part.strip() for part in logger.split(",") if part.strip()]
        return [str(item) for item in logger]

    def _plain_config(self) -> Any:
        try:
            from omegaconf import OmegaConf

            return OmegaConf.to_container(
                self.config,
                resolve=True,
                throw_on_missing=False,
            )
        except Exception:
            return self.config

    def _init_tracking(self) -> None:
        if self.rank != 0:
            return
        if "wandb" not in self._logger_names():
            return

        wandb_cfg = self._config_get("trainer", "wandb", default=None)
        if wandb_cfg is not None:
            wandb_entity = self._nested_get(wandb_cfg, "entity", None)
            wandb_mode = self._nested_get(wandb_cfg, "mode", None)
            wandb_dir = self._nested_get(wandb_cfg, "dir", None)
            if wandb_entity:
                os.environ["WANDB_ENTITY"] = str(wandb_entity)
            if wandb_mode:
                os.environ["WANDB_MODE"] = str(wandb_mode)
            if wandb_dir:
                os.environ["WANDB_DIR"] = str(wandb_dir)
                os.makedirs(os.path.expanduser(str(wandb_dir)), exist_ok=True)

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "trainer.logger includes 'wandb', but wandb is not installed."
            ) from exc

        self._wandb_run = wandb.init(
            project=str(
                self._config_get("trainer", "project_name", default="flexmdream")
            ),
            name=str(
                self._config_get("trainer", "experiment_name", default="fsdp")
            ),
            config=self._plain_config(),
        )

    def _log_metrics(self, metrics: Dict[str, float], *, step: int) -> None:
        if self.rank != 0:
            return
        if "console" in self._logger_names():
            print({"step": int(step), **metrics}, flush=True)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=int(step))

    def _finish_tracking(self) -> None:
        if self.rank != 0 or self._wandb_run is None:
            return
        self._wandb_run.finish()
        self._wandb_run = None

    @staticmethod
    def _nested_get(container: Any, key: str, default: Any = None) -> Any:
        if isinstance(container, dict):
            return container.get(key, default)
        if hasattr(container, "get"):
            value = container.get(key, default)
            return default if value is None else value
        return getattr(container, key, default)

    def save_checkpoint(self, step: int) -> None:
        """Save a checkpoint from the FSDP-wrapped FlexMDM model.

        The FlexMDM wrapper is not a PreTrainedModel, so the save splits the
        full state dict into two pieces: the Dream-Coder backbone is written
        in the standard HF format via `backbone.save_pretrained`, and the
        FlexMDM-specific modules (`time_mlp`, `temb_mods`, `insertion_head`)
        are written to a companion `flexmdm_extras.pt` alongside it. The
        AdamW optimizer state is gathered to rank 0 via the FSDP optim API
        and saved as `optimizer_state.pt` so a later run can resume the
        Adam moments exactly.

        Architecture: the synchronous path is the FSDP gather (a collective
        all ranks must join). After the gather, rank 0 forks a subprocess
        that owns the heavy disk writes, then returns. A subprocess (rather
        than a thread) is required because torch.save's pickle path holds
        the Python GIL for the duration of large-tensor writes, which would
        starve the main thread of CPU time and block its NCCL kernel
        launches — that previously caused all-gather collectives on other
        ranks to time out at the 600s NCCL watchdog. The subprocess has its
        own interpreter, so its disk work cannot stall the parent.
        """
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        # Reap the previous save subprocess. This bounds peak CPU memory on
        # rank 0 (we never gather a fresh ~50GB state dict while the previous
        # one is still in a child holding it via copy-on-write) and surfaces
        # any error from the previous save loudly here, in the main thread.
        if self._pending_save_pid is not None:
            self._await_pending_save()

        output_dir = os.path.join(
            os.path.expanduser(str(self.config.trainer.default_local_dir)),
            f"global_step_{int(step)}",
        )
        rng_dir = os.path.join(output_dir, "rng_state")
        # Every rank creates the dirs (idempotent). This replaces the
        # previous inter-rank barriers whose only job was to ensure rank 0
        # had created output_dir / rng_dir before non-rank-0 writers
        # touched them.
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(rng_dir, exist_ok=True)

        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_config = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            self.fsdp_model,
            StateDictType.FULL_STATE_DICT,
            state_config,
            optim_state_config,
        ):
            model_state = self.fsdp_model.state_dict()
            optim_state = FSDP.optim_state_dict(self.fsdp_model, self.optimizer)

        # Each rank writes its own RNG state directly — small (a few KB),
        # parallel across ranks.
        torch.save(
            self._capture_rng_state(),
            os.path.join(rng_dir, f"rank_{self.rank}.pt"),
        )

        if self.rank == 0:
            # Snapshot the bits of trainer state we need before forking; the
            # main thread (parent) is free to mutate epoch/lr_scheduler the
            # moment we return. The child sees these values through fork's
            # copy-on-write — but post-fork mutations in the parent would not
            # propagate, so capturing here is purely for clarity.
            lr_state = (
                None
                if self.lr_scheduler is None
                else self.lr_scheduler.state_dict()
            )
            current_epoch = int(self._current_epoch)
            next_batch_in_epoch = int(self._next_batch_in_epoch)
            global_step = int(step)

            # Fork. The child writes disk and exits; the parent records the
            # pid and returns immediately to keep training NCCL collectives
            # un-stalled.
            # Flush parent stdio before fork: any buffered bytes would
            # otherwise be duplicated to disk by both parent and child.
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            pid = os.fork()
            if pid == 0:
                # Child: must NOT touch CUDA (parent's CUDA context is
                # inherited but cannot be safely used here). Pure CPU/disk
                # work, then os._exit so we skip all atexit/finalizer paths
                # that the parent owns.
                #
                # Redirect child stdout/stderr to a per-step log via dup2.
                # Two reasons:
                #   1) If the child crashes silently (e.g., the HF tokenizer
                #      Rust pool deadlocking after fork — what killed
                #      job 9647803 at step 10500), we still get a trace.
                #   2) Insulates the parent's slurm-captured stdio from any
                #      damage caused by sharing fds across the fork. Job
                #      9647803's parent stopped flushing to slurm .out/.err
                #      after step 10500's fork.
                try:
                    log_fd = os.open(
                        os.path.join(output_dir, "save_child.log"),
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        0o644,
                    )
                    os.dup2(log_fd, 1)
                    os.dup2(log_fd, 2)
                    os.close(log_fd)
                except Exception:
                    pass
                # Disable every thread pool we know touches the save path.
                # Forking from a multi-threaded parent is hazardous: the child
                # inherits each library's lock state but not the threads that
                # would release them, so the first call into the same library
                # can deadlock. Empirically this killed job 9647803 (HF
                # tokenizers Rust pool) and is the most likely cause of
                # job 9798214's step_16500 stall (safetensors uses Rayon).
                # Forcing every pool single-threaded means there is no
                # inherited lock to deadlock against.
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["RAYON_NUM_THREADS"] = "1"
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
                exit_code = 0
                try:
                    self._write_main_checkpoint_files(
                        output_dir,
                        model_state,
                        optim_state,
                        global_step,
                        current_epoch,
                        next_batch_in_epoch,
                        lr_state,
                    )
                except BaseException:
                    traceback.print_exc()
                    sys.stderr.flush()
                    exit_code = 1
                finally:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                    os._exit(exit_code)
            else:
                self._pending_save_pid = pid
                self._pending_save_dir = output_dir
                self._pending_save_step = global_step

    def _await_pending_save(self, timeout_sec: int = 1800) -> None:
        """Block until the previous save subprocess exits, with a timeout.

        Behavior is **non-raising** for every failure mode: clean exit
        returns silently; non-zero exit, signal kill, or a child stuck
        past `timeout_sec` (in which case the child is SIGKILL'd) all log
        loudly and write a `_save_failed.txt` marker into the partial
        checkpoint dir, then return.

        The run survives one bad save by design. With `training_state.pt`
        written last (see `_write_main_checkpoint_files`), a poisoned dir
        is automatically excluded by AUTO_RESUME's commit-marker gate, so
        a future resume will pick up from the last *complete* checkpoint
        rather than from a partial one.

        Why a timeout: the previous unbounded `os.waitpid(pid, 0)` was the
        single point at which a hung child could deadlock rank 0 forever
        (job 9798214's step_16500). With ranks 1-15 then NCCL-timing out
        at the next collective, a single transient stall destroyed the
        entire run. Bounding the wait converts that into a logged
        skipped-save and lets training continue.
        """
        pid = self._pending_save_pid
        pending_dir = self._pending_save_dir
        pending_step = self._pending_save_step
        self._pending_save_pid = None
        self._pending_save_dir = None
        self._pending_save_step = None
        if pid is None:
            return

        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                wpid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # Already reaped externally; lose the exit code but proceed.
                return
            if wpid == pid:
                if os.WIFEXITED(status):
                    code = os.WEXITSTATUS(status)
                    if code == 0:
                        return
                    self._record_save_failure(
                        pending_dir,
                        pending_step,
                        f"child pid {pid} exited with code {code}",
                    )
                    return
                if os.WIFSIGNALED(status):
                    sig = os.WTERMSIG(status)
                    self._record_save_failure(
                        pending_dir,
                        pending_step,
                        f"child pid {pid} killed by signal {sig}",
                    )
                    return
                # Stopped/continued — shouldn't happen for our children.
                return
            if time.monotonic() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                self._record_save_failure(
                    pending_dir,
                    pending_step,
                    f"child pid {pid} hung >{timeout_sec}s; SIGKILL'd",
                )
                return
            time.sleep(2.0)

    def _record_save_failure(
        self,
        output_dir: Optional[str],
        step: Optional[int],
        reason: str,
    ) -> None:
        """Log a save failure and drop a marker file into the partial dir.

        Stdout may be unreliable on long runs (slurm pipe buffering quirks
        observed in 9798214), so the marker file is the durable record.
        AUTO_RESUME does not need to look at it — its commit-marker gate
        on training_state.pt already excludes the partial dir — but it is
        invaluable for post-mortem.
        """
        msg = f"[save] step {step}: {reason}; checkpoint skipped"
        try:
            print(msg, flush=True)
        except Exception:
            pass
        if output_dir is None:
            return
        try:
            with open(os.path.join(output_dir, "_save_failed.txt"), "w") as f:
                f.write(reason + "\n")
        except Exception:
            pass

    def _write_main_checkpoint_files(
        self,
        output_dir: str,
        model_state: Dict[str, torch.Tensor],
        optim_state: Dict[str, Any],
        step: int,
        epoch: int,
        next_batch_in_epoch: int,
        lr_scheduler_state: Optional[Dict[str, Any]],
    ) -> None:
        """Run inside rank 0's forked save subprocess. No collectives, no CUDA.

        save_pretrained with an explicit state_dict only reads the model's
        config (immutable) and writes the tensors we hand it; it does not
        touch the live FSDP-wrapped parameters. The child inherits the
        config object via copy-on-write but never writes through it, so this
        is safe even though the parent's FSDP/CUDA state is technically
        still attached.
        """
        wrapper = self.fsdp_model.module
        backbone_prefix = "backbone."
        backbone_state = {
            key[len(backbone_prefix):]: value
            for key, value in model_state.items()
            if key.startswith(backbone_prefix)
        }
        extras_state = {
            key: value
            for key, value in model_state.items()
            if not key.startswith(backbone_prefix)
        }
        # Order: training_state.pt is written LAST as a commit marker.
        # AUTO_RESUME's gate checks training_state.pt presence + non-zero
        # size, so its existence guarantees every prior file is fully
        # written. A child that dies (or is timeout-killed) mid-write
        # leaves a directory that AUTO_RESUME automatically excludes,
        # and the run's next save attempt is unaffected.
        #
        # `.stage` is a tiny file rewritten before each step. If a future
        # save hangs again, `cat <ckpt>/.stage` tells us which call stalled.
        def _stage(name: str) -> None:
            try:
                with open(os.path.join(output_dir, ".stage"), "w") as f:
                    f.write(name + "\n")
            except Exception:
                pass

        _stage("backbone")
        wrapper.backbone.save_pretrained(
            output_dir, state_dict=backbone_state
        )
        # save_pretrained rewrites auto_map.AutoConfig to a relative
        # path while leaving auto_map.AutoModel pointing at the HF hub
        # repo (Dream-org/Dream-Coder-v0-Base-7B--modeling_dream.DreamModel).
        # On resume, AutoConfig then loads configuration_dream.py from
        # the checkpoint dir as `transformers_modules.global_step_N.
        # configuration_dream.DreamConfig`, while AutoModel loads it
        # from the hub as `transformers_modules.Dream-org.Dream-Coder-
        # v0-Base-7B.<rev>.configuration_dream.DreamConfig`. Same code,
        # different module path, so transformers' strict config_class
        # check rejects them (job 9719942 died this way at startup).
        # Force AutoConfig to share the hub prefix so both come from
        # the same module path.
        _stage("auto_map_fix")
        try:
            cfg_path = os.path.join(output_dir, "config.json")
            with open(cfg_path) as f:
                cfg_json = json.load(f)
            am = cfg_json.get("auto_map", {})
            am_model = am.get("AutoModel", "")
            if "--" in am_model:
                hub_prefix = am_model.split("--", 1)[0] + "--"
                for k in ("AutoConfig", "AutoModel"):
                    v = am.get(k)
                    if v and "--" not in v:
                        am[k] = hub_prefix + v
                cfg_json["auto_map"] = am
                with open(cfg_path, "w") as f:
                    json.dump(cfg_json, f, indent=2)
        except Exception:
            traceback.print_exc()

        _stage("optimizer")
        torch.save(
            optim_state,
            os.path.join(output_dir, "optimizer_state.pt"),
        )

        _stage("tokenizer")
        self.tokenizer.save_pretrained(output_dir)

        _stage("flexmdm_extras")
        torch.save(
            extras_state,
            os.path.join(output_dir, "flexmdm_extras.pt"),
        )

        # COMMIT MARKER. Must be last.
        _stage("training_state")
        torch.save(
            {
                "global_step": step,
                "epoch": epoch,
                "next_batch_in_epoch": next_batch_in_epoch,
                "lr_scheduler": lr_scheduler_state,
            },
            os.path.join(output_dir, "training_state.pt"),
        )
        _stage("done")

    def _capture_rng_state(self) -> Dict[str, Any]:
        import random as _random

        import numpy as _np

        return {
            "python": _random.getstate(),
            "numpy": _np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state(self.device)
                if torch.cuda.is_available()
                else None
            ),
        }

    def _restore_rng_state(self, state: Dict[str, Any]) -> None:
        import random as _random

        import numpy as _np

        _random.setstate(state["python"])
        _np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state, self.device)

    def load_checkpoint(self, resume_from: str) -> None:
        """Restore optimizer state, LR scheduler, and global_step from a ckpt.

        Backbone weights and FlexMDM extras must already be loaded into the
        model before this is called — see `train_fsdp.build_model_and_tokenizer`,
        which handles those pre-FSDP-wrap so `sync_module_states` propagates
        the loaded values to non-rank-0 ranks.

        The optimizer file is optional: checkpoints saved before optimizer
        state was being persisted will simply leave the Adam moments fresh.
        `training_state.pt` (global_step + lr_scheduler) is required.
        """
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )

        resume_from = os.path.expanduser(str(resume_from))

        optim_path = os.path.join(resume_from, "optimizer_state.pt")
        if os.path.isfile(optim_path):
            state_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            optim_state_config = FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                state_config,
                optim_state_config,
            ):
                full_osd = (
                    torch.load(optim_path, map_location="cpu")
                    if self.rank == 0
                    else None
                )
                sharded = FSDP.optim_state_dict_to_load(
                    model=self.fsdp_model,
                    optim=self.optimizer,
                    optim_state_dict=full_osd,
                )
                self.optimizer.load_state_dict(sharded)
            if self.rank == 0:
                print(
                    f"[resume] loaded optimizer state from {optim_path}",
                    flush=True,
                )
        else:
            if self.rank == 0:
                print(
                    f"[resume] no optimizer_state.pt in {resume_from} — Adam "
                    "moments start fresh. Future checkpoints will include it.",
                    flush=True,
                )

        state_path = os.path.join(resume_from, "training_state.pt")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"resume_from={resume_from!r} missing training_state.pt"
            )
        training_state = torch.load(
            state_path, map_location="cpu", weights_only=False
        )
        self.global_step = int(training_state["global_step"])
        # epoch/next_batch_in_epoch are optional for back-compat with
        # checkpoints saved before dataloader-position tracking existed.
        self._resume_epoch = int(training_state.get("epoch", 0))
        self._resume_batch_index = int(
            training_state.get("next_batch_in_epoch", 0)
        )
        if (
            self.lr_scheduler is not None
            and training_state.get("lr_scheduler") is not None
        ):
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        if self.rank == 0:
            scheduler_status = (
                "restored"
                if training_state.get("lr_scheduler") is not None
                else "fresh"
            )
            print(
                f"[resume] global_step set to {self.global_step}; "
                f"epoch={self._resume_epoch}, "
                f"next_batch_in_epoch={self._resume_batch_index}; "
                f"lr_scheduler={scheduler_status}",
                flush=True,
            )

        rng_path = os.path.join(
            resume_from, "rng_state", f"rank_{self.rank}.pt"
        )
        if os.path.isfile(rng_path):
            rng_state = torch.load(
                rng_path, map_location="cpu", weights_only=False
            )
            self._restore_rng_state(rng_state)
            if self.rank == 0:
                print(
                    f"[resume] restored per-rank RNG state from "
                    f"{os.path.dirname(rng_path)}/",
                    flush=True,
                )
        else:
            if self.rank == 0:
                print(
                    f"[resume] no rng_state in {resume_from} — RNG continues "
                    "from set_seed init. Future checkpoints will include it.",
                    flush=True,
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


__all__ = [
    "FlexMDMFSDPTrainer",
    "FlexMDMProcess",
    "poisson_loss",
]

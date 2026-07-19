# -------------------------------------------------------
# bidirectional Transformer (no time-embedding)
# based on QWen architecture
# -------------------------------------------------------

from dataclasses import dataclass, replace
from typing import Optional, Tuple
from model.utils import RotaryEmbedding, RMSNorm, apply_rope
from torch.backends.cuda import sdp_kernel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# -------------------------------------------------------
# TransformerConfig
# -------------------------------------------------------
@dataclass
class MDMConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    max_position: int = 1024
    rms_norm_eps: float = 1e-6
    tie_lm_head: bool = False
    bias_qkv: bool = True
    dropout: float = 0.0
    causal: bool = False
    arm_init: Optional[str] = None
    predict_next_token: bool = False   # True when initialized from ARM
    input_type: str = "discrete" # "discrete" or "continuous"
    input_dim: Optional[int] = None # only used if input_type == "continuous"
    input_projection: bool = False # whether to project continuous inputs to hidden size
    return_last_hidden_states: bool = False # whether to return last hidden states instead of lm
    apply_final_norm: bool = True # if False, final_norm is Identity (skip RMSNorm before lm_head)
    segment_pooling: str = "mean" # "mean" or "eos_mean"; used by LP-MDM encoder

    def head_dim(self) -> int:
        # dimension for each head
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads


# -------------------------------------------------------
# MLP
# -------------------------------------------------------
class SwiGLU(nn.Module):
    """
    SwiGLU: (xW1) ⊗ swish(xW2) @ W3
    Shapes:
      in=hidden, hidden=intermediate_size
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = F.silu(self.w2(x))
        out = self.w3(a * b)
        return self.dropout(out)


# -------------------------------------------------------
# Qwen attention
# -------------------------------------------------------
class QwenAttention(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        H = config.num_attention_heads
        K = config.num_kv_heads
        d = config.hidden_size
        Hd = config.head_dim()
        assert H % K ==0, "num_attention_heads must be divisible by num_kv_heads"
        self.kv_repeats = H // K

        # qkv projection
        self.q_proj = nn.Linear(d, H * Hd, bias=config.bias_qkv)
        self.k_proj = nn.Linear(d, K * Hd, bias=config.bias_qkv)
        self.v_proj = nn.Linear(d, K * Hd, bias=config.bias_qkv)
        self.o_proj = nn.Linear(H * Hd, d, bias=config.bias_qkv)

        # rope
        self.rope = RotaryEmbedding(Hd, max_position=config.max_position)

        self.dropout_p = config.dropout

    def _shape(self, x, B, L, H, Hd):
        return x.view(B, L, H, Hd)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        # L = seqlen of new tokens being fed in this call
        B, L, _ = x.shape

        H = self.config.num_attention_heads
        K = self.config.num_kv_heads
        Hd = self.config.head_dim()

        # qkv projection and reshape
        q = self._shape(self.q_proj(x), B, L, H, Hd)
        k = self._shape(self.k_proj(x), B, L, K, Hd)
        v = self._shape(self.v_proj(x), B, L, K, Hd)

        # RoPE — applied at absolute positions [position_offset, position_offset + L)
        cos, sin = self.rope(L, position_offset=position_offset)
        q, k = apply_rope(q, k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Append cached KV from earlier steps
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        causal = bool(getattr(self.config, "causal", False))
        is_causal_param = False
        attn_mask_eff = attn_mask
        if causal:
            if past_kv is None:
                # standard square causal attention
                is_causal_param = True
            elif L > 1:
                # incremental with multiple new tokens: build explicit causal mask
                past_len = past_kv[0].shape[2]
                row = torch.arange(L, device=x.device).unsqueeze(1)
                col = torch.arange(past_len + L, device=x.device).unsqueeze(0)
                mask = col <= (past_len + row)
                attn_mask_eff = mask.unsqueeze(0).unsqueeze(0)
            # When L == 1 with past_kv: single query attends to every key, no mask needed.

        out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask_eff,
                    dropout_p=(self.dropout_p if self.training else 0.0),
                    is_causal=is_causal_param,
                    enable_gqa=(H != K),
            )  # (B, H, L, Hd)

        out = out.transpose(1, 2).contiguous().view(B, L, H * Hd)
        out = self.o_proj(out)
        if use_cache:
            return out, new_kv
        return out


# -------------------------------------------------------
# Transformeer Block
# -------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.attn = QwenAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, dropout = config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        # pre-norm
        h_in = self.attn_norm(x)
        if use_cache:
            attn_out, new_kv = self.attn(
                h_in,
                attn_mask=attn_mask,
                past_kv=past_kv,
                use_cache=True,
                position_offset=position_offset,
            )
            h = x + attn_out
            h = h + self.mlp(self.mlp_norm(h))
            return h, new_kv
        h = x + self.attn(h_in, attn_mask=attn_mask)
        h = h + self.mlp(self.mlp_norm(h))
        return h


# -------------------------------------------------------
# Full model
# -------------------------------------------------------
class MDMTransformer(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        if config.input_type == "discrete":
            self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            if config.input_projection:
                self.emb = nn.Linear(config.input_dim, config.hidden_size)
            else:
                self.emb = nn.Identity()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        if config.apply_final_norm:
            self.final_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        else:
            self.final_norm = nn.Identity()
        if not config.return_last_hidden_states:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
            if config.tie_lm_head:
                self.lm_head.weight = self.emb.weight
        else:
            self.lm_head = nn.Identity()

    def forward_embeds(self, inputs_embeds: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if inputs_embeds.dim() != 3:
            raise ValueError(
                f"inputs_embeds must have shape (B, L, D), got {tuple(inputs_embeds.shape)}"
            )

        x = inputs_embeds

        # forward pass
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        # final layer
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

    def forward(self, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if self.config.input_type == "discrete" and input_ids.dim() != 2:
            raise ValueError(
                f"discrete input_ids must have shape (B, L), got {tuple(input_ids.shape)}"
            )
        if self.config.input_type == "continuous" and input_ids.dim() != 3:
            raise ValueError(
                f"continuous inputs must have shape (B, L, D), got {tuple(input_ids.shape)}"
            )

        # token embeddings or continuous input projection
        x = self.emb(input_ids)
        return self.forward_embeds(x, attn_mask=attn_mask)

    def forward_step(
        self,
        inputs_embeds: torch.Tensor,
        past_kvs: Optional[list] = None,
        position_offset: int = 0,
    ):
        """Incremental forward with KV caching for causal transformers.

        ``inputs_embeds`` are only the newly fed positions (B, L_new, D).
        ``past_kvs`` is a list of (k, v) per layer (or None for the first call).
        ``position_offset`` is the absolute position of ``inputs_embeds[:, 0]``.
        Returns ``(logits, new_past_kvs)``.
        """
        if not getattr(self.config, "causal", False):
            raise ValueError("forward_step requires a causal transformer.")
        x = inputs_embeds
        new_past_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer(
                x,
                past_kv=past_kv,
                use_cache=True,
                position_offset=position_offset,
            )
            new_past_kvs.append(new_kv)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x, new_past_kvs

class LatentMDMTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert config.input_type == "continuous", "LatentMDMTransformer only supports continuous inputs"
        self.prompt_emb = nn.Embedding(config.vocab_size, config.hidden_size)

class EncodeTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert config.input_type == "discrete", "EncodeTransformer only supports discrete inputs"

class DecodeTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert config.input_type == "discrete", "DecodeTransformer only supports discrete inputs"
        assert config.causal, "DecodeTransformer must be initialized with causal=True"
    def forward(self, input_ids: torch.Tensor, planner_condition: torch.Tensor):
        B, L = input_ids.shape
        if planner_condition.dim() == 2:
            planner_condition = planner_condition.unsqueeze(1)
        if planner_condition.shape[:2] != (B, 1):
            raise ValueError(
                "planner_condition must have shape (B, D) or (B, 1, D), "
                f"got {tuple(planner_condition.shape)} for batch {B}"
            )

        # token embeddings
        x = self.emb(input_ids)
        
        x = torch.cat([planner_condition, x], dim=1) # (B, 1+L, d)

        # forward pass
        for layer in self.layers:
            x = layer(x)

        # final layer
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

class CombinedTransformer(nn.Module):
    def __init__(
        self,
        encoder_config: MDMConfig,
        planner_config: MDMConfig,
        decoder_config: MDMConfig,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        assert encoder_config.return_last_hidden_states, "LP-MDM encoder must return hidden states"
        assert planner_config.return_last_hidden_states, "LP-MDM planner must return hidden states"
        assert not decoder_config.return_last_hidden_states, "LP-MDM decoder must return logits"
        use_cls = str(getattr(encoder_config, "segment_pooling", "mean")) == "cls"
        if use_cls:
            self.cls_token_id = int(encoder_config.vocab_size)
            encoder_config = replace(
                encoder_config, vocab_size=encoder_config.vocab_size + 1
            )
            if tie_embeddings:
                planner_config = replace(
                    planner_config, vocab_size=planner_config.vocab_size + 1
                )
                decoder_config = replace(
                    decoder_config, vocab_size=decoder_config.vocab_size + 1
                )
        else:
            self.cls_token_id = None
        self.encoder = EncodeTransformer(encoder_config)
        self.planner = LatentMDMTransformer(planner_config)
        self.decoder = DecodeTransformer(decoder_config)
        if tie_embeddings:
            if not (
                encoder_config.vocab_size == planner_config.vocab_size == decoder_config.vocab_size
                and encoder_config.hidden_size == planner_config.hidden_size == decoder_config.hidden_size
            ):
                raise ValueError(
                    "tie_embeddings requires encoder, planner, and decoder to share vocab_size "
                    "and hidden_size."
                )
            self.planner.prompt_emb.weight = self.encoder.emb.weight
            self.decoder.emb.weight = self.encoder.emb.weight
            if isinstance(self.decoder.lm_head, nn.Linear):
                self.decoder.lm_head.weight = self.encoder.emb.weight
        self.mask_token = nn.Parameter(torch.randn(planner_config.hidden_size) * 0.02)
        if encoder_config.hidden_size != planner_config.hidden_size:
            self.encoder_to_planner = nn.Linear(encoder_config.hidden_size, planner_config.hidden_size)
        else:
            self.encoder_to_planner = nn.Identity()

if __name__ == "__main__":
    print("Building a MDM Transformer...")

    cfg = MDMConfig(
        vocab_size = 2000,
        hidden_size = 256,
        intermediate_size = 1024,
        num_layers = 4,
        num_attention_heads = 4,
        num_kv_heads = 2)
        
    model = MDMTransformer(cfg)
    B, L = 2, 16
    
    # random input
    x = torch.randint(0, cfg.vocab_size, (B, L))
    print(f"Input shape: {x.shape}")

    # model forward pass
    logits = model(x)
    print(f"Logits shape: {logits.shape}")

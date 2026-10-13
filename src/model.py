"""SPT model used by the final IJOE event-split experiment.

The module names remain compatible with checkpoints produced by the original
research implementation. No pretrained checkpoint is distributed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SPTConfig:
    input_dim: int = 4
    embed_dim: int = 128
    heads: int = 8
    blocks: int = 2
    dropout: float = 0.01
    attention_type: str = "offset"
    qk_norm: bool = True
    num_points: int = 512
    feature_indices: tuple[int, ...] = (1, 2, 3, 4)

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """Small RMSNorm implementation with timm-compatible state-dict keys."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


class Attention(nn.Module):
    """Multi-head attention compatible with timm's ``Attention`` parameters."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        weights = (q * self.scale) @ k.transpose(-2, -1)
        weights = self.attn_drop(weights.softmax(dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    """MLP with timm-compatible ``fc1``/``fc2`` parameter names."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class SPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attention_type: str, qk_norm: bool) -> None:
        super().__init__()
        if attention_type not in {"offset", "residual"}:
            raise ValueError("attention_type must be 'offset' or 'residual'")
        self.attention_type = attention_type
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention_type == "residual":
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        residual = x - self.attn(self.norm1(x))
        residual = residual + self.mlp(self.norm2(residual))
        return x + residual


class SPT(nn.Module):
    """Permutation-invariant regression from sparse spectral points to Hs."""

    def __init__(self, config: SPTConfig | None = None) -> None:
        super().__init__()
        self.config = config or SPTConfig()
        cfg = self.config
        if cfg.input_dim != len(cfg.feature_indices):
            raise ValueError("input_dim must match feature_indices")
        self.embedding = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embed_dim // 2),
            nn.LayerNorm(cfg.embed_dim // 2),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim // 2, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            SPTBlock(cfg.embed_dim, cfg.heads, cfg.attention_type, cfg.qk_norm)
            for _ in range(cfg.blocks)
        )
        concat_dim = cfg.embed_dim * (cfg.blocks + 1)
        self.regressor = nn.Sequential(
            nn.Linear(concat_dim * 2, concat_dim),
            nn.LayerNorm(concat_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(concat_dim, 1),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected [batch, points, {self.config.input_dim}], got {tuple(x.shape)}")
        features = [self.embedding(x)]
        for block in self.blocks:
            features.append(block(features[-1]))
        local = torch.cat(features, dim=-1)
        global_max = local.max(dim=1).values
        global_avg = local.mean(dim=1)
        global_feature = torch.cat((global_max, global_avg), dim=-1)
        return {"local": local, "global_max": global_max, "global_avg": global_avg,
                "global": global_feature}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.forward_features(x)["global"]).squeeze(-1)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[SPT, dict]:
    """Load either the packaged training checkpoint or a plain state dict."""
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    cfg = SPTConfig(
        input_dim=int(saved_cfg.get("input_dim", 4)),
        embed_dim=int(saved_cfg.get("embed_dim", 128)),
        heads=int(saved_cfg.get("heads", 8)),
        blocks=int(saved_cfg.get("blocks", saved_cfg.get("block_num", 2))),
        dropout=float(saved_cfg.get("dropout", 0.01)),
        attention_type=str(saved_cfg.get("attention_type", "offset")),
        qk_norm=bool(saved_cfg.get("qk_norm", True)),
        num_points=int(saved_cfg.get("num_points", 512)),
        feature_indices=tuple(saved_cfg.get("feature_indices", [1, 2, 3, 4])),
    )
    model = SPT(cfg).to(device)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return model, payload

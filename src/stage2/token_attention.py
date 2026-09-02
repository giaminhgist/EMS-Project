"""Optional fused-token normative branch (guide 05 §13, contracts §16).

Enabled only when ``model.bank_mode == "trial_and_fused_token"`` and the bank
provides fused token arrays. Serial topology::

    R0 -> LocalMHA1 + residual + LayerNorm = H1
       -> spatial bridge + gated residual + LayerNorm = Hb
       -> LocalMHA2 + residual + LayerNorm = H2
       -> token attention pooling -> d_token
       -> [z_trial, d_token] fusion -> z_extended

Attention 1 and 2 do not share parameters, and Attention 2 consumes the bridge
output ``Hb``. Query tokens are pre-fusion and bank tokens are post-fusion, so
token comparisons go through independent learned projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_H, GRID_W = 12, 16
N_TOKENS = GRID_H * GRID_W
WINDOW = 3  # 3x3 local bank window
WINDOW_SIZE = WINDOW * WINDOW
CENTER_OFFSET = WINDOW_SIZE // 2  # the self/center cell


class TokenProjections(nn.Module):
    """Independent query/bank projections for tokens (guide §13.1)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.bank_mean = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.bank_sigma = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.reliability = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))

    def project_query(self, heatmap_tokens: torch.Tensor) -> torch.Tensor:
        """Pre-fusion query token projection ``Q``."""
        return self.query(heatmap_tokens)

    def project_bank(
        self, mu_token: torch.Tensor, sigma_token: torch.Tensor, count: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Post-fusion bank projections ``(N_mu, N_ctx, rho_token)``."""
        n_mu = self.bank_mean(mu_token)
        n_ctx = self.bank_sigma(torch.log(sigma_token.clamp_min(1e-12)))
        neg_log_sigma_mean = (-torch.log(sigma_token.clamp_min(1e-12))).mean(dim=-1, keepdim=True)
        log_count = torch.log1p(count.to(sigma_token.dtype))[:, None, None].expand_as(
            neg_log_sigma_mean
        )
        rho_token = torch.sigmoid(
            self.reliability(torch.cat([neg_log_sigma_mean, log_count], dim=-1))
        )  # [N,192,1]
        return n_mu, n_ctx, rho_token

    def forward(
        self,
        heatmap_tokens: torch.Tensor,  # [N,192,D] pre-fusion
        mu_token: torch.Tensor,  # [N,192,D] post-fusion bank mean
        sigma_token: torch.Tensor,  # [N,192,D] bank sigma
        count: torch.Tensor,  # [N]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_tok = self.project_query(heatmap_tokens)
        n_mu, n_ctx, rho_token = self.project_bank(mu_token, sigma_token, count)
        return q_tok, n_mu, n_ctx, rho_token


class AlignedTokenRelation(nn.Module):
    """Per-patch aligned relation: 514-dim features -> MLP -> ``R0`` (§13.2)."""

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.feature_dim = 4 * dim + 2  # Q, N_mu, |Q-N_mu|, Q*N_mu, cosine, rho
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(
        self,
        q_tok: torch.Tensor,
        n_mu: torch.Tensor,
        cosine: torch.Tensor,
        rho_token: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [q_tok, n_mu, (q_tok - n_mu).abs(), q_tok * n_mu, cosine, rho_token], dim=-1
        )
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"token relation features must be [{self.feature_dim}], got {features.shape[-1]}"
            )
        return self.mlp(features)  # R0 [N,192,D]


def build_window_index() -> tuple[torch.Tensor, torch.Tensor]:
    """Padded 3x3 neighbor indices and validity mask for every grid position.

    Row-major ``t = row * 16 + column``; the window never wraps between the
    last column of one row and the first column of the next.
    """
    index = torch.zeros(N_TOKENS, WINDOW_SIZE, dtype=torch.int64)
    valid = torch.zeros(N_TOKENS, WINDOW_SIZE, dtype=torch.bool)
    offsets = [
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
    ]
    for t in range(N_TOKENS):
        r, c = divmod(t, GRID_W)
        for j, (dr, dc) in enumerate(offsets):
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                index[t, j] = rr * GRID_W + cc
                valid[t, j] = True
    return index, valid


class LocalBankMHA(nn.Module):
    """Multi-head cross-attention restricted to each query's 3x3 bank window."""

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.register_buffer("window_index", build_window_index()[0])
        self.register_buffer("window_valid", build_window_index()[1])
        self.rel_pos_bias = nn.Parameter(torch.zeros(heads, WINDOW_SIZE))

    def forward(
        self, x: torch.Tensor, bank_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``x [N,192,D]`` attends over local windows of ``bank_tokens``.

        Returns ``(output [N,192,D], attention weights [N,4,192,9])``.
        """
        n = x.shape[0]
        q = self.q_proj(x).reshape(n, N_TOKENS, self.heads, self.head_dim)  # [N,192,H,Dh]
        k = self.k_proj(bank_tokens).reshape(n, N_TOKENS, self.heads, self.head_dim)
        v = self.v_proj(bank_tokens).reshape(n, N_TOKENS, self.heads, self.head_dim)
        k_local = k[:, self.window_index]  # [N,192,9,H,Dh]
        v_local = v[:, self.window_index]
        logits = torch.einsum("nthd,ntkhd->nhtk", q, k_local) * (self.head_dim ** -0.5)
        logits = logits + self.rel_pos_bias[None, :, None, :]
        valid = self.window_valid[None, None, :, :].expand(n, 1, N_TOKENS, WINDOW_SIZE)
        logits = logits.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # [N,4,192,9]
        out = torch.einsum("nhtk,ntkhd->nthd", weights, v_local)  # [N,192,4,Dh]
        out = out.reshape(n, N_TOKENS, self.dim)
        return self.out_proj(out), weights


class SpatialBridge(nn.Module):
    """``[N,192,128] -> [N,128,12,16] -> depthwise Conv3x3 -> pointwise
    128->256 -> GELU/dropout -> pointwise 256->128 -> [N,192,128]`` (§13.4)."""

    def __init__(self, dim: int = 128, expansion: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.expand = nn.Conv2d(dim, expansion, 1)
        self.project = nn.Conv2d(expansion, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        spatial = x.transpose(1, 2).reshape(n, x.shape[-1], GRID_H, GRID_W)
        spatial = self.dwconv(spatial)
        spatial = self.dropout(F.gelu(self.expand(spatial)))
        spatial = self.project(spatial)
        return spatial.reshape(n, x.shape[-1], N_TOKENS).transpose(1, 2)


class TokenAttentionPooler(nn.Module):
    """Attention pooling of ``H2`` to one token ``d_token [N,128]`` (§13.5)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = self.score(self.norm(tokens)).softmax(dim=1)  # [N,192,1]
        return (weights * tokens).sum(dim=1)  # [N,D]


class TokenTrialFusion(nn.Module):
    """``[z_trial, d_token] [N,256] -> Linear(256,128) + GELU -> residual
    z_trial -> LayerNorm -> z_extended`` (§13.5)."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_trial: torch.Tensor, d_token: torch.Tensor) -> torch.Tensor:
        fused = F.gelu(self.proj(torch.cat([z_trial, d_token], dim=-1)))
        return self.norm(fused + z_trial)


class FusedTokenBranch(nn.Module):
    """Serial local-bank cross-attention over aligned token relations.

    ``layers=1`` is the ``single_token_attention`` ablation (pool the bridge
    output; Attention 2 is absent). ``bridge="identity"`` is the
    ``no_spatial_bridge`` ablation (the gated residual remains).
    """

    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1,
                 layers: int = 2, bridge: str = "residual_dwconv_ffn"):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError("token attention layers must be 1 or 2")
        self.layers = layers
        self.projections = TokenProjections(dim)
        self.token_relation = AlignedTokenRelation(dim)
        self.mha1 = LocalBankMHA(dim, heads)  # Attention 1 — distinct storage
        if layers == 2:
            self.mha2 = LocalBankMHA(dim, heads)  # Attention 2 — distinct storage
        else:
            self.mha2 = None
        if bridge == "residual_dwconv_ffn":
            self.bridge = SpatialBridge(dim, dropout=dropout)
        elif bridge == "identity":
            self.bridge = nn.Identity()
        else:
            raise ValueError(f"unsupported token spatial bridge {bridge!r}")
        self.bridge_gate = nn.Parameter(torch.tensor(0.1))  # gated residual
        self.norm1 = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.pooler = TokenAttentionPooler(dim)
        self.fusion = TokenTrialFusion(dim)

    def forward(
        self,
        heatmap_tokens: torch.Tensor,  # H [N,192,128]
        mu_token: torch.Tensor,  # [N,192,128]
        sigma_token: torch.Tensor,  # [N,192,128]
        count: torch.Tensor,  # [N]
        z_trial: torch.Tensor,  # [N,128]
        *,
        keep_full_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        q_tok, n_mu, n_ctx, rho_token = self.projections(
            heatmap_tokens, mu_token, sigma_token, count
        )
        token_cosine = (F.normalize(q_tok, dim=-1) * F.normalize(n_mu, dim=-1)).sum(
            dim=-1, keepdim=True
        ).clamp(-1.0, 1.0)  # [N,192,1]

        r0 = self.token_relation(q_tok, n_mu, token_cosine, rho_token)

        mha1_out, attn1 = self.mha1(r0, n_mu)
        h1 = self.norm1(r0 + mha1_out)
        bridged = self.bridge(h1)
        h_b = self.norm_b(h1 + self.bridge_gate * bridged)  # gated residual
        if self.layers == 2:
            h2, attn_last = self.mha2(h_b, n_mu)  # Attention 2 consumes the bridge output
            h2 = self.norm2(h_b + h2)
        else:
            h2, attn_last = h_b, attn1  # single layer: pool Hb

        d_token = self.pooler(h2)
        z_extended = self.fusion(z_trial, d_token)

        # Reduced normalized semantic map: mean attention to the center cell,
        # reliability-weighted mismatch (contracts §16.3, guide §13.6).
        center_attention = attn_last.mean(dim=1)[:, :, CENTER_OFFSET]  # [N,192]
        map_raw = center_attention * rho_token.squeeze(-1) * (1.0 - token_cosine.squeeze(-1))
        map_min = map_raw.min(dim=1, keepdim=True).values
        map_max = map_raw.max(dim=1, keepdim=True).values
        map_norm = (map_raw - map_min) / (map_max - map_min).clamp_min(1e-6)

        return {
            "z_extended": z_extended,
            "token_attention_weights": attn_last if keep_full_attention else None,
            "token_omega": center_attention,
            "token_map_flat": map_norm,
            "token_cosine": token_cosine.squeeze(-1),
            "token_rho": rho_token,
            "Q": q_tok,
            "N_mu": n_mu,
            "N_ctx": n_ctx,
        }

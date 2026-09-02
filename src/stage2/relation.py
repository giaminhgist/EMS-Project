"""Explicit normative relation between query and trial bank (guide 05 §6-§7,
contracts §14.3-§14.4).

Query tokens are pre-fusion and the trial bank is post-fusion, so raw
subtraction or raw cosine between them is invalid. All comparisons go through
learned query and bank projections, and the relation feature vector is built
explicitly from aligned projected quantities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryProjection(nn.Module):
    """``LayerNorm -> Linear(128,128)``."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q0: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(q0))


class BankMeanAdapter(nn.Module):
    """``LayerNorm -> Linear(128,128)`` over the trial bank mean."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, mu_raw: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(mu_raw))


class BankSigmaAdapter(nn.Module):
    """Uncertainty context from bank sigma and trial count (guide §6).

    ``concat(LayerNorm(log_sigma), log_count) [N,129] ->
    Linear(129,256) -> GELU -> Linear(256,128)``.
    """

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, sigma_raw: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """``sigma_raw [N,D]``, ``count [N]`` -> ``uncertainty_context [N,D]``."""
        log_sigma = torch.log(sigma_raw.clamp_min(1e-12))
        log_count = torch.log1p(count.to(log_sigma.dtype)).unsqueeze(-1)  # [N,1]
        return self.proj(torch.cat([self.norm(log_sigma), log_count], dim=-1))


class ReliabilityHead(nn.Module):
    """Per-trial reliability ``rho in (0,1)`` (guide §6).

    ``[mean(-log_sigma), log_count] [N,2] -> MLP -> sigmoid [N,1]``.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, sigma_raw: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """``sigma_raw [N,D]``, ``count [N]`` -> ``rho [N,1]``."""
        neg_log_sigma_mean = (-torch.log(sigma_raw.clamp_min(1e-12))).mean(dim=-1, keepdim=True)
        log_count = torch.log1p(count.to(sigma_raw.dtype)).unsqueeze(-1)
        return torch.sigmoid(self.mlp(torch.cat([neg_log_sigma_mean, log_count], dim=-1)))


class ComparatorHead(nn.Module):
    """Scalar relation/comparator score consumed by the bank-rank loss."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, z_trial: torch.Tensor) -> torch.Tensor:
        return self.score(z_trial).squeeze(-1)  # [N]


class TrialRelation(nn.Module):
    """Explicit aligned relation (guide §7).

    Feature vector ``[N,770]``: q, n_mu, uncertainty_context, q-n_mu,
    abs(q-n_mu), q*n_mu (768) + cos(q,n_mu) + rho (2). Relation MLP:
    ``Linear(770,256) -> GELU -> Dropout -> Linear(256,128)``, residual query
    projection shortcut, then LayerNorm.
    """

    def __init__(self, dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.feature_dim = 6 * dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm_out = nn.LayerNorm(dim)

    def build_features(
        self, q: torch.Tensor, n_mu: torch.Tensor, uncertainty_context: torch.Tensor,
        cosine: torch.Tensor, rho: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                q,
                n_mu,
                uncertainty_context,
                q - n_mu,
                (q - n_mu).abs(),
                q * n_mu,
                cosine,  # [N,1]
                rho,  # [N,1]
            ],
            dim=-1,
        )

    def forward(
        self, q: torch.Tensor, n_mu: torch.Tensor, uncertainty_context: torch.Tensor,
        cosine: torch.Tensor, rho: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_features(q, n_mu, uncertainty_context, cosine, rho)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"relation features must be [{self.feature_dim}], got {features.shape[-1]}"
            )
        return self.norm_out(self.mlp(features) + q)


def row_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity ``[N,1]`` with an epsilon floor (no NaNs)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)


class TrialRelationBlock(nn.Module):
    """End-to-end trial relation: projections, adapters, reliability, MLP.

    Inputs are raw gathered bank rows (``mu_raw/sigma_raw/count``) and the
    pooled query ``q0``. Returns ``(z_trial [N,128], cosine [N,1], rho [N,1],
    q [N,128], n_mu [N,128], uncertainty_context [N,128])``.

    ``active=False`` is the ``no_bank`` ablation: every bank-dependent block
    receives deterministic neutral values (zero mean, unit sigma, unit count)
    so the relation width is preserved while no path reads real bank values.
    """

    def __init__(self, dim: int = 128, hidden: int = 256, dropout: float = 0.1,
                 active: bool = True):
        super().__init__()
        self.active = bool(active)
        self.query_projection = QueryProjection(dim)
        self.bank_mean_adapter = BankMeanAdapter(dim)
        self.bank_sigma_adapter = BankSigmaAdapter(dim)
        self.reliability = ReliabilityHead()
        self.relation = TrialRelation(dim, hidden, dropout)
        self.comparator = ComparatorHead(dim)

    def _resolve_bank(self, q0: torch.Tensor, mu_raw: torch.Tensor | None,
                      sigma_raw: torch.Tensor | None, count: torch.Tensor | None
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.active:
            n = q0.shape[0]
            return (
                torch.zeros_like(q0),
                torch.ones_like(q0),
                torch.ones(n, dtype=torch.int64, device=q0.device),
            )
        if mu_raw is None or sigma_raw is None or count is None:
            raise ValueError("bank tensors are required when bank features are active")
        return mu_raw, sigma_raw, count

    def project(self, q0: torch.Tensor, mu_raw: torch.Tensor, sigma_raw: torch.Tensor,
                count: torch.Tensor) -> dict[str, torch.Tensor]:
        """Projections + reliability for one bank; relation MLP is not applied."""
        return self.project_with_query(self.query_projection(q0), mu_raw, sigma_raw, count)

    def forward(self, q0: torch.Tensor, mu_raw: torch.Tensor | None = None,
                sigma_raw: torch.Tensor | None = None,
                count: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Full relation for one bank: projections + relation MLP + comparator."""
        mu_raw, sigma_raw, count = self._resolve_bank(q0, mu_raw, sigma_raw, count)
        out = self.project(q0, mu_raw, sigma_raw, count)
        return self.forward_with_query(out["q"], mu_raw, sigma_raw, count)

    def forward_with_query(self, q: torch.Tensor, mu_raw: torch.Tensor | None,
                           sigma_raw: torch.Tensor | None,
                           count: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Relation/comparator for a second bank reusing an existing query
        projection (wrong-bank runs must not reproject ``q0``)."""
        mu_raw, sigma_raw, count = self._resolve_bank(q, mu_raw, sigma_raw, count)
        out = self.project_with_query(q, mu_raw, sigma_raw, count)
        out["z_trial"] = self.relation(
            out["q"], out["n_mu"], out["uncertainty_context"], out["cosine"], out["rho"]
        )
        out["comparator"] = self.comparator(out["z_trial"])
        return out

    def project_with_query(self, q: torch.Tensor, mu_raw: torch.Tensor,
                           sigma_raw: torch.Tensor, count: torch.Tensor) -> dict[str, torch.Tensor]:
        """Bank-side projections + reliability for a supplied query projection."""
        n_mu = self.bank_mean_adapter(mu_raw)
        uncertainty_context = self.bank_sigma_adapter(sigma_raw, count)
        rho = self.reliability(sigma_raw, count)
        return {
            "q": q,
            "n_mu": n_mu,
            "uncertainty_context": uncertainty_context,
            "rho": rho,
            "cosine": row_cosine(q, n_mu),
        }

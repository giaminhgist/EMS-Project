"""Configuration and ablation framework (guide 06): the named-ablation
registry, resolution order (base -> overlay -> CLI overrides -> runtime
fields), overlay-diff validation against declared changes, the wrong-bank
permutation and the global-bank statistics, and a bulk dry-run CLI.

An ablation changes only its declared factor; the overlay diff rejects any
extra or missing change. Multiple named ablations in one run are rejected
unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    ConfigError,
    Stage2Config,
    coerce_cli_value,
    config_hash,
    deep_merge,
    load_yaml_dict,
)

BASE_CONFIG_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "stage2" / "base.yaml"
ABLATION_DIR = Path(__file__).resolve().parents[2] / "configs" / "stage2" / "ablations"


@dataclass(frozen=True)
class AblationSpec:
    """Provenance record for one named ablation (guide 06 §4)."""

    name: str
    scientific_question: str
    declared_changes: tuple[str, ...]
    required_bank_capabilities: tuple[str, ...]  # trial_bank | fused_token_bank | none
    forbidden_with: tuple[str, ...] = ()
    interpretation: str = ""
    is_negative_control: bool = False
    reference: str = "base"  # configuration this run family is compared against


ABLATIONS: dict[str, AblationSpec] = {
    "base": AblationSpec(
        name="base",
        scientific_question="Full trial-bank model",
        declared_changes=(),
        required_bank_capabilities=("trial_bank",),
        interpretation="Reference configuration",
    ),
    "no_bank": AblationSpec(
        name="no_bank",
        scientific_question="Does normative information add value?",
        declared_changes=("model.bank_features_active",),
        required_bank_capabilities=("none",),
        interpretation="Negative control: relation width preserved, bank neutralized",
        is_negative_control=True,
    ),
    "wrong_stimulus_bank": AblationSpec(
        name="wrong_stimulus_bank",
        scientific_question="Is stimulus matching important?",
        declared_changes=("model.wrong_bank_permutation",),
        required_bank_capabilities=("trial_bank",),
        interpretation="Negative control: one fixed category-preserving derangement",
        is_negative_control=True,
    ),
    "global_bank": AblationSpec(
        name="global_bank",
        scientific_question="Is per-stimulus normalization important?",
        declared_changes=("model.global_bank",),
        required_bank_capabilities=("trial_bank",),
        interpretation="Negative control: one global training-HC bank vector",
        is_negative_control=True,
    ),
    "random_encoder": AblationSpec(
        name="random_encoder",
        scientific_question="Do transferred Stage-1 heatmap features add value?",
        declared_changes=("model.encoder_random_init",),
        required_bank_capabilities=("trial_bank",),
        interpretation="Transfer negative control: same architecture, seeded random frozen weights",
        is_negative_control=True,
    ),
    "unfreeze_last_block": AblationSpec(
        name="unfreeze_last_block",
        scientific_question="Does limited encoder adaptation help?",
        declared_changes=("model.encoder_unfreeze_last_block", "loss.lambda_anchor"),
        required_bank_capabilities=("trial_bank",),
        interpretation="Stage 2C: final residual block trainable at 0.1x LR with anchor",
    ),
    "mean_query_pooling": AblationSpec(
        name="mean_query_pooling",
        scientific_question="Does learned patch pooling help?",
        declared_changes=("model.query_pooling",),
        required_bank_capabilities=("trial_bank",),
    ),
    "no_category_balance": AblationSpec(
        name="no_category_balance",
        scientific_question="Does category balancing prevent count dominance?",
        declared_changes=("model.category_balanced_attention",),
        required_bank_capabilities=("trial_bank",),
    ),
    "mean_subject_pooling": AblationSpec(
        name="mean_subject_pooling",
        scientific_question="Does inter-stimulus contextualization help?",
        declared_changes=("model.subject_transformer_layers",),
        required_bank_capabilities=("trial_bank",),
    ),
    "no_match_loss": AblationSpec(
        name="no_match_loss",
        scientific_question="Is explicit query/bank alignment useful?",
        declared_changes=("loss.lambda_match", "optimization.alignment_epochs"),
        required_bank_capabilities=("trial_bank",),
    ),
    "no_aux_loss": AblationSpec(
        name="no_aux_loss",
        scientific_question="Does faithful additive evidence supervision help?",
        declared_changes=("loss.lambda_aux",),
        required_bank_capabilities=("trial_bank",),
    ),
    "no_consistency_loss": AblationSpec(
        name="no_consistency_loss",
        scientific_question="Does subset consistency improve robustness?",
        declared_changes=("loss.lambda_cons", "subsets.enabled"),
        required_bank_capabilities=("trial_bank",),
    ),
    "no_attention_entropy": AblationSpec(
        name="no_attention_entropy",
        scientific_question="Does early attention regularization help?",
        declared_changes=("loss.lambda_entropy",),
        required_bank_capabilities=("trial_bank",),
    ),
    "token_bank_serial_attention": AblationSpec(
        name="token_bank_serial_attention",
        scientific_question="Does local normative semantic structure add value?",
        declared_changes=("model.bank_mode", "bank.require_fused_token_bank"),
        required_bank_capabilities=("fused_token_bank",),
    ),
    "single_token_attention": AblationSpec(
        name="single_token_attention",
        scientific_question="Are two serial token layers necessary?",
        declared_changes=(
            "model.bank_mode", "bank.require_fused_token_bank", "model.token_attention_layers"
        ),
        required_bank_capabilities=("fused_token_bank",),
        reference="token_bank_serial_attention",
    ),
    "no_spatial_bridge": AblationSpec(
        name="no_spatial_bridge",
        scientific_question="Does spatial token propagation add value?",
        declared_changes=(
            "model.bank_mode", "bank.require_fused_token_bank", "model.token_spatial_bridge"
        ),
        required_bank_capabilities=("fused_token_bank",),
        reference="token_bank_serial_attention",
    ),
    "same_space_heat_bank": AblationSpec(
        name="same_space_heat_bank",
        scientific_question="Is direct same-encoder token deviation sufficient?",
        declared_changes=("model.bank_mode", "bank.require_heatmap_token_bank"),
        required_bank_capabilities=("heatmap_token_bank",),
        interpretation="Not runnable: heatmap-token banks were not built in Phase 0",
    ),
}


# ------------------------------------------------------------- helpers


def flatten_config(raw: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a configuration dict to dotted keys (values are scalars/paths)."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_config(value, f"{path}."))
        else:
            out[path] = value
    return out


# Keys that identify the run/fold rather than the scientific configuration;
# excluded from the scientific overlay diff but still stored.
RUNTIME_DERIVED_EXCLUDED = ("ablation",)


def compute_overlay_diff(
    base_raw: dict[str, Any], resolved_raw: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """``{dotted_key: (base_value, resolved_value)}`` for changed keys."""
    base_flat = flatten_config(base_raw)
    resolved_flat = flatten_config(resolved_raw)
    diff: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(base_flat) | set(resolved_flat)):
        if key in RUNTIME_DERIVED_EXCLUDED:
            continue
        if base_flat.get(key) != resolved_flat.get(key):
            diff[key] = (base_flat.get(key), resolved_flat.get(key))
    return diff


def build_wrong_bank_permutation(
    category_ids: list[int], *, seed: int, fold: int
) -> dict[int, int]:
    """One fixed, category-preserving derangement of stimulus indices.

    Deterministic in ``(seed, fold)``, never maps a stimulus to itself when
    its category has at least two members, and is identical for every subject
    in the run (guide 06 §7.2).
    """
    n = len(category_ids)
    members_by_category: dict[int, list[int]] = {}
    for s, c in enumerate(category_ids):
        members_by_category.setdefault(int(c), []).append(int(s))
    rng = np.random.default_rng(int(seed) * 31 + int(fold) * 7)
    permutation: dict[int, int] = {}
    for members in members_by_category.values():
        members = sorted(members)
        if len(members) < 2:
            permutation[members[0]] = members[0]  # no alternative within category
            continue
        offset = int(rng.integers(1, len(members)))
        for i, s in enumerate(members):
            permutation[s] = members[(i + offset) % len(members)]
    return {s: permutation[s] for s in range(n)}


def compute_global_bank_stats(
    mu: torch.Tensor, sigma: torch.Tensor, count: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Count-weighted global mean and pooled diagonal variance (guide 06 §7.3).

    ``mu [S,D]``, ``sigma [S,D]``, ``count [S]``; float64 accumulation;
    ``gsigma^2 = sum_s c_s (sigma_s^2 + mu_s^2) / C - gmu^2``.
    """
    c = count.to(torch.float64)
    total = c.sum()
    if total <= 0:
        raise ValueError("global bank requires at least one contributing sample")
    gmu = (mu.to(torch.float64) * c.unsqueeze(-1)).sum(dim=0) / total
    second_moment = ((sigma.to(torch.float64) ** 2 + mu.to(torch.float64) ** 2)
                     * c.unsqueeze(-1)).sum(dim=0) / total
    gsigma_sq = (second_moment - gmu ** 2).clamp_min(1e-12)
    return (
        gmu.to(torch.float32),
        gsigma_sq.sqrt().to(torch.float32),
        int(total.item()),
    )


def global_bank_checksum(gmu: torch.Tensor, gsigma: torch.Tensor, gcount: int) -> str:
    """Stable checksum of one global-bank entry (guide 06 §7.3)."""
    import hashlib

    payload = {
        "gmu": [float(x) for x in gmu.tolist()],
        "gsigma": [float(x) for x in gsigma.tolist()],
        "gcount": gcount,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------- resolution


@dataclass
class ResolvedConfig:
    config: Stage2Config
    raw: dict[str, Any]
    config_hash: str
    spec: AblationSpec
    changed_keys: list[str]
    diff_entries: list[str]  # "key: base -> resolved"


def resolve_ablation_config(
    base_path: Path | str,
    ablation: str,
    *,
    overrides: dict[str, Any] | None = None,
    fold: int | None = None,
) -> ResolvedConfig:
    """base.yaml -> one named ablation overlay -> CLI overrides -> fold."""
    if "+" in ablation or "," in ablation:
        raise ConfigError(
            f"multiple named ablations ({ablation!r}) are not allowed by default; "
            f"a factorial experiment requires an explicit request"
        )
    if ablation not in ABLATIONS:
        raise ConfigError(
            f"unknown ablation {ablation!r}; registered: {sorted(ABLATIONS)}"
        )
    spec = ABLATIONS[ablation]
    base_raw = load_yaml_dict(base_path)
    overlay_path = ABLATION_DIR / f"{ablation}.yaml"
    if ablation != "base":
        if not overlay_path.is_file():
            raise ConfigError(f"ablation overlay missing: {overlay_path}")
        overlay_raw = load_yaml_dict(overlay_path)
        unknown = set(overlay_raw) - set(base_raw)
        if unknown:
            raise ConfigError(f"{overlay_path}: unknown top-level fields: {sorted(unknown)}")
        overlaid = deep_merge(base_raw, overlay_raw)
    else:
        overlaid = deep_merge(base_raw, {})

    # The scientific overlay diff covers base -> base+overlay only; explicit
    # CLI overrides are user-requested changes applied afterwards and are not
    # part of the declared factor (guide §3 resolution order).
    resolved = deep_merge(overlaid, {})
    overrides = overrides or {}
    for key, value in overrides.items():
        if isinstance(value, str):
            value = coerce_cli_value(value)
        node: dict[str, Any] = resolved
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ConfigError(f"override {key!r} traverses a non-mapping field")
            node = node[part]
        node[parts[-1]] = value

    resolved["ablation"] = ablation
    if fold is not None:
        resolved["fold"] = int(fold)

    cfg = Stage2Config.from_dict(resolved, source=f"ablation={ablation}")
    diff = compute_overlay_diff(base_raw, overlaid)
    changed = set(diff)
    declared = set(spec.declared_changes)
    if changed != declared:
        extra = sorted(changed - declared)
        missing = sorted(declared - changed)
        raise ConfigError(
            f"ablation {ablation!r} changes {sorted(changed)} but declares {sorted(declared)}"
            + (f"; undeclared: {extra}" if extra else "")
            + (f"; declared but unchanged: {missing}" if missing else "")
        )
    return ResolvedConfig(
        config=cfg,
        raw=resolved,
        config_hash=config_hash(resolved),
        spec=spec,
        changed_keys=sorted(changed),
        diff_entries=[f"{k}: {v0} -> {v1}" for k, (v0, v1) in sorted(diff.items())],
    )


def validate_bank_capabilities(spec: AblationSpec, bank_store: Any) -> None:
    """Required bank capabilities against the loaded bank store (guide 06 §7.6)."""
    for capability in spec.required_bank_capabilities:
        if capability == "trial_bank":
            continue  # always present
        if capability == "fused_token_bank" and not getattr(bank_store, "has_token_banks", False):
            raise ConfigError(
                f"ablation {spec.name!r} requires fused token banks; fold "
                f"{bank_store.fold} does not provide them"
            )
        if capability == "heatmap_token_bank":
            raise ConfigError(
                f"ablation {spec.name!r} is not runnable: heatmap-token banks were not "
                f"built (Phase 0 approved trial + fused-token banks only)"
            )


# ------------------------------------------------------------- bulk dry run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation framework checks (no training)")
    parser.add_argument(
        "--config", default=str(BASE_CONFIG_DEFAULT)
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dry-run-all",
        action="store_true",
        help="resolve every registered ablation and run one synthetic batch forward/backward",
    )
    parser.add_argument("--ablation", default=None, help="resolve and print one ablation")
    return parser


def _synthetic_batch(device: str = "cpu", category_ids_panel: list[int] | None = None):
    """One small synthetic batch: 4 subjects, 8 valid trials each.

    ``category_ids_panel`` must match the bank feature manifest so
    same-category negative sampling stays consistent with the real data.
    """
    from .collate import Stage2Batch

    torch.manual_seed(2026)
    b = 4
    heatmaps = torch.randn(b, 100, 3, 48, 64)
    heatmaps[:, :2] = heatmaps[:, :2].abs()
    mask = torch.zeros(b, 100, dtype=torch.bool)
    if category_ids_panel is None:
        category_ids_panel = [i % 4 for i in range(100)]
    # Two valid trials from every category so all aggregation paths and the
    # same-category negative sampling are exercised.
    members: dict[int, list[int]] = {}
    for s, c in enumerate(category_ids_panel):
        members.setdefault(int(c), []).append(s)
    valid_slots = [s for c in sorted(members) for s in members[c][:2]]
    mask[:, valid_slots] = True
    category_ids = torch.tensor(category_ids_panel, dtype=torch.int64).expand(b, -1).clone()
    return Stage2Batch(
        subject_ids=("000", "001", "002", "003"),
        labels=torch.tensor([0.0, 0.0, 1.0, 1.0]),
        heatmaps=heatmaps,
        trial_mask=mask,
        stimulus_indices=torch.arange(100).expand(b, -1).clone(),
        category_ids=category_ids,
        bank_split_ids=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        trial_uids=tuple(
            tuple(f"{i}_{s}" if bool(mask[i, s]) else None for s in range(100)) for i in range(b)
        ),
    ).to_device(device)


def dry_run_all(cfg: Stage2Config, device: str) -> None:
    from .bank import NormativeBankStore
    from .losses import compute_stage2_losses, generate_subset_masks
    from .model import Stage2Model

    bank_store = NormativeBankStore(cfg, cfg.fold, device=device)
    print(f"fold {cfg.fold} | regime {bank_store.evaluation_regime} | "
          f"token banks present: {bank_store.has_token_banks}")
    for name, spec in ABLATIONS.items():
        print("----")
        print(f"name: {name}")
        print(f"  scientific question: {spec.scientific_question}")
        print(f"  reference configuration: {spec.reference}")
        print(f"  declared changes: {list(spec.declared_changes)}")
        print(f"  required bank capabilities: {list(spec.required_bank_capabilities)}")
        try:
            resolved = resolve_ablation_config(BASE_CONFIG_DEFAULT, name, fold=cfg.fold)
            validate_bank_capabilities(spec, bank_store)
        except ConfigError as exc:
            print(f"  status: NOT RUNNABLE — {exc}")
            continue
        run_cfg = resolved.config
        model = Stage2Model(run_cfg, bank_store, device=device)
        report = model.parameter_report()
        batch = _synthetic_batch(
            device, category_ids_panel=bank_store.feature_manifest.category_id.astype(int).tolist()
        )
        subset_masks = None
        if run_cfg.subsets.enabled:
            subset_masks = generate_subset_masks(
                trial_mask=batch.trial_mask, category_ids=batch.category_ids,
                subject_ids=list(batch.subject_ids), seed=run_cfg.seed, fold=cfg.fold,
                epoch=0, train=True,
            )
        result = model(batch, "train", subset_masks=subset_masks)
        enc = model.encode_trials(batch, "train")
        match_inputs = model.matching_inputs(batch, enc=enc, epoch=0)
        anchor_current, anchor_stage1 = model.transferred_encoder.anchor_vectors()
        if run_cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
            anchor_current = anchor_stage1 = None
        losses = compute_stage2_losses(
            loss_cfg=run_cfg.loss, subsets_cfg=run_cfg.subsets, labels=batch.labels,
            full=result.full, subsets=result.subsets, match_inputs=match_inputs,
            anchor_current=anchor_current, anchor_stage1=anchor_stage1, epoch=0,
        )
        losses.total.backward()
        finite = torch.isfinite(losses.total).item()
        grads_ok = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        print(f"  changed dotted keys: {resolved.changed_keys}")
        print(f"  trainable parameters: {report['trainable']} | frozen: {report['frozen']}")
        print(f"  input shapes: heatmaps {tuple(batch.heatmaps.shape)} | "
              f"valid trials {batch.n_valid_trials}")
        print(f"  output shapes: main_logit {tuple(result.full.main_logit.shape)} | "
              f"subject_embedding {tuple(result.full.subject_embedding.shape)}")
        print(f"  total loss: {float(losses.total.detach())} | "
              f"cls {float(losses.cls.detach()):.4f} | aux {float(losses.aux.detach()):.4f} | "
              f"match {float(losses.match.detach()):.4f} | cons {float(losses.cons.detach()):.4f} | "
              f"entropy {float(losses.entropy.detach()):.4f} | "
              f"anchor {float(losses.anchor.detach()):.4f}")
        print(f"  finite forward/backward: {finite and grads_ok}")
    print("----")
    print("bulk ablation dry run complete")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.ablation and not args.dry_run_all:
            resolved = resolve_ablation_config(args.config, args.ablation, fold=args.fold)
            print(f"ablation: {resolved.config.ablation}")
            print(f"reference: {resolved.spec.reference}")
            print(f"changed keys:")
            for entry in resolved.diff_entries:
                print(f"  {entry}")
            print(f"config hash: {resolved.config_hash}")
            return 0
        cfg = Stage2Config.from_yaml(args.config)
        cfg = Stage2Config.from_dict({**cfg.to_dict(), "fold": args.fold})
        if not args.dry_run_all:
            print("use --dry-run-all to check every registered ablation, "
                  "or --ablation NAME to resolve one")
            return 0
        dry_run_all(cfg, args.device)
        return 0
    except (ConfigError, ValueError, OSError) as exc:
        print(f"ablation check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

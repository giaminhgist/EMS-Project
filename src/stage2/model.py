"""Stage-2 subject classifier: transferred encoder -> query pooler -> explicit
normative relation -> category-balanced subject aggregation (guide 05, §16-§17).

``forward`` encodes the batch once (frozen Stage-1 encoder) and reruns only the
subject aggregation for the full panel and each supplied subset mask. The
optional fused-token branch activates only under
``model.bank_mode == "trial_and_fused_token"`` and fails at initialization when
the bank has no token arrays. No training loop or checkpoint lifecycle lives
here — this phase provides the model and a dry-run CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .bank import FULL_BANK_ID, NormativeBankStore
from .collate import collate_subject_samples, flatten_valid_trials
from .config import ConfigError, Stage2Config
from .contracts import (
    D_MODEL,
    N_STIMULI,
    EncodedTrials,
    Stage2ForwardResult,
    Stage2MatchInputs,
    Stage2ForwardOutput,
)
from .dataset import Stage2SubjectDataset
from .pooling import build_query_pooler
from .relation import TrialRelationBlock
from .sampler import BalancedSubjectBatchSampler
from .subject_aggregation import SubjectAggregator
from .token_attention import FusedTokenBranch
from .transferred_encoder import TransferredHeatmapEncoder, TransferredEncoderError

from torch.utils.data import DataLoader

log = logging.getLogger("stage2.model")


class Stage2ModelError(ValueError):
    pass


class Stage2Model(nn.Module):
    """One outer fold's Stage-2 model bound to its normative bank store."""

    def __init__(
        self,
        cfg: Stage2Config,
        bank_store: NormativeBankStore,
        *,
        device: str = "cpu",
    ):
        super().__init__()
        if bank_store.fold != cfg.fold:
            raise Stage2ModelError(
                f"bank store fold {bank_store.fold} != config fold {cfg.fold}"
            )
        self.cfg = cfg
        self.bank_store = bank_store
        self.fold = cfg.fold
        self.device = device

        registry_entry = bank_store.registry_entry
        self.transferred_encoder = TransferredHeatmapEncoder(
            registry_entry["checkpoint"],
            expected_sha256=str(registry_entry["sha256"]),
            fold=cfg.fold,
            freeze=cfg.model.freeze_encoder,
            device=device,
            random_init=cfg.model.encoder_random_init,
            init_seed=cfg.seed,
        )
        if cfg.model.encoder_unfreeze_last_block:
            self.transferred_encoder.unfreeze_last_block()
        self.pooler = build_query_pooler(cfg.model.query_pooling, D_MODEL)
        self.relation = TrialRelationBlock(
            D_MODEL, hidden=cfg.model.relation_hidden, dropout=cfg.model.dropout,
            active=cfg.model.bank_features_active,
        )
        self.aggregator = SubjectAggregator(
            dim=D_MODEL,
            heads=cfg.model.attention_heads,
            ffn_dim=cfg.model.subject_transformer_ffn,
            dropout=cfg.model.dropout,
            transformer_layers=cfg.model.subject_transformer_layers,
            category_balanced=cfg.model.category_balanced_attention,
        )

        self.bank_mode = cfg.model.bank_mode
        self.bank_features_active = cfg.model.bank_features_active
        self.global_bank = cfg.model.global_bank
        self.wrong_bank_permutation: dict[int, int] | None = None
        if cfg.model.wrong_bank_permutation:
            from .ablations import build_wrong_bank_permutation

            category_ids = bank_store.feature_manifest.category_id.astype(int).tolist()
            self.wrong_bank_permutation = build_wrong_bank_permutation(
                category_ids, seed=cfg.seed, fold=cfg.fold
            )
        self.token_branch: FusedTokenBranch | None = None
        if self.bank_mode == "trial_and_fused_token":
            if not bank_store.has_token_banks:
                raise Stage2ModelError(
                    f"model.bank_mode={self.bank_mode!r} but fold {self.fold} has no "
                    f"fused token banks — fail at initialization"
                )
            self.token_branch = FusedTokenBranch(
                D_MODEL, heads=cfg.model.attention_heads, dropout=cfg.model.dropout,
                layers=cfg.model.token_attention_layers,
                bridge=cfg.model.token_spatial_bridge,
            )

        # Category membership for same-category negative sampling.
        fm = bank_store.feature_manifest
        self._category_members: dict[int, list[int]] = {}
        for cat in range(4):
            self._category_members[cat] = fm[fm.category_id.astype(int) == cat][
                "stimulus_index"
            ].astype(int).tolist()

        self.to(device)

    # ------------------------------------------------------------------ mode

    def train(self, mode: bool = True) -> "Stage2Model":
        super().train(mode)
        # Wrapper invariant: a frozen encoder can never re-enter train mode.
        self.transferred_encoder.train(mode)
        return self

    # ------------------------------------------------------------------ encode

    def encode_trials(
        self,
        batch: Any,
        split: str,
        *,
        debug_token_attention: bool = False,
    ) -> EncodedTrials:
        """Encode valid trials once: encoder + pooler + relation (+ token branch)."""
        batch.validate()
        flat = flatten_valid_trials(batch)
        heatmaps = flat.heatmaps.to(self.device)
        heatmap_tokens = self.transferred_encoder(heatmaps)  # [N,192,128]
        patch_attention, q0 = self.pooler(heatmap_tokens)

        if self.bank_features_active:
            stimulus_indices = flat.stimulus_indices
            if self.wrong_bank_permutation is not None:
                stimulus_indices = torch.tensor(
                    [self.wrong_bank_permutation[int(s)] for s in stimulus_indices.tolist()],
                    dtype=torch.int64,
                )
            gathered = self.bank_store.gather_trials(
                stimulus_indices=stimulus_indices,
                subject_slots=flat.subject_slots,
                subject_bank_ids=batch.bank_split_ids,
                split=split,
                device=self.device,
            )
            if self.global_bank:
                self._apply_global_bank(gathered)
            rel = self.relation(
                q0, gathered.mu_trial, gathered.sigma_trial, gathered.count_trial
            )
            bank_ids = gathered.bank_ids
            mu_token = gathered.mu_token
            sigma_token = gathered.sigma_token
            count_trial = gathered.count_trial
        else:
            # no_bank: no path reads any bank value.
            rel = self.relation(q0)
            bank_ids = torch.full(
                (flat.subject_slots.numel(),), FULL_BANK_ID, dtype=torch.int64, device=self.device
            )
            mu_token = sigma_token = count_trial = None

        enc = EncodedTrials(
            batch_size=batch.n_subjects,
            subject_slots=flat.subject_slots.to(self.device),
            stimulus_slots=flat.stimulus_slots.to(self.device),
            category_ids=flat.category_ids.to(self.device),
            trial_mask=batch.trial_mask.to(self.device),
            category_ids_panel=batch.category_ids.to(self.device),
            heatmap_tokens=heatmap_tokens,
            patch_attention=patch_attention,
            q0=q0,
            q=rel["q"],
            n_mu=rel["n_mu"],
            uncertainty_context=rel["uncertainty_context"],
            rho=rel["rho"],
            cosine=rel["cosine"],
            z_trial=rel["z_trial"],
            comparator=rel["comparator"],
            bank_ids=bank_ids,
        )
        if self.token_branch is not None:
            token_out = self.token_branch(
                heatmap_tokens,
                mu_token,
                sigma_token,
                count_trial,
                rel["z_trial"],
                keep_full_attention=debug_token_attention,
            )
            enc.z_extended = token_out["z_extended"]
            enc.token_attention_weights = token_out["token_attention_weights"]
            enc.token_omega = token_out["token_omega"]
            enc.token_map_flat = token_out["token_map_flat"]
            enc.token_cosine = token_out["token_cosine"]
            enc.token_rho = token_out["token_rho"]
            enc.Q = token_out["Q"]
            enc.N_mu = token_out["N_mu"]
        return enc

    def _apply_global_bank(self, gathered: Any) -> None:
        """`global_bank` ablation: replace gathered rows by one count-weighted
        global mean/variance entry per bank id (guide 06 §7.3)."""
        from .ablations import compute_global_bank_stats

        store = self.bank_store
        for bank_id in sorted({int(x) for x in gathered.bank_ids.tolist()}):
            rows = gathered.bank_ids == bank_id
            if bank_id == FULL_BANK_ID:
                mu = store.mu_trial
                sigma = store.sigma_trial
                count = store.count_trial
            else:
                arrays = store.split_tensors[int(bank_id)]
                mu = arrays["mu_trial"]
                sigma = arrays["sigma_trial"]
                count = arrays["count_trial"]
            gmu, gsigma, gcount = compute_global_bank_stats(mu, sigma, count)
            gathered.mu_trial[rows] = gmu
            gathered.sigma_trial[rows] = gsigma
            gathered.count_trial[rows] = gcount
            if gathered.mu_token is not None and store.has_token_banks:
                if bank_id == FULL_BANK_ID:
                    tmu = store.mu_token
                    tsigma = store.sigma_token
                else:
                    arrays = store.split_tensors[int(bank_id)]
                    tmu = arrays["mu_token"]
                    tsigma = arrays["sigma_token"]
                tgmu, tgsigma, _ = compute_global_bank_stats(
                    tmu.reshape(100, -1), tsigma.reshape(100, -1),
                    store.count_trial if bank_id == FULL_BANK_ID else arrays["count_trial"],
                )
                gathered.mu_token[rows] = tgmu.reshape(192, 128)
                gathered.sigma_token[rows] = tgsigma.reshape(192, 128)

    def aggregate_subject(
        self, enc: EncodedTrials, subset_mask: torch.Tensor | None = None
    ) -> Stage2ForwardOutput:
        """Rerunnable subject aggregation over one effective trial mask."""
        return self.aggregator(enc, subset_mask)

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        batch: Any,
        split: str = "train",
        *,
        subset_masks: dict[str, torch.Tensor] | None = None,
        debug_token_attention: bool = False,
    ) -> Stage2ForwardResult:
        """Encode once; aggregate the full panel and each subset mask."""
        enc = self.encode_trials(batch, split, debug_token_attention=debug_token_attention)
        full = self.aggregate_subject(enc)
        subsets: dict[str, Stage2ForwardOutput] = {}
        for name, mask in (subset_masks or {}).items():
            subsets[name] = self.aggregate_subject(enc, mask)
        return Stage2ForwardResult(full=full, subsets=subsets)

    # ----------------------------------------------------------------- matching

    def _sample_negative_stimuli(
        self, stimulus_indices: torch.Tensor, category_ids: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        """Deterministic same-category negative per trial; never the positive.

        Trials whose category has a single stimulus are returned with the
        positive index itself; the caller records them as skipped.
        """
        import numpy as np

        n = stimulus_indices.numel()
        negatives = torch.empty_like(stimulus_indices)
        rng = np.random.default_rng(self.cfg.seed * 31 + self.fold * 7 + int(epoch))
        stimulus_np = stimulus_indices.cpu().numpy()
        category_np = category_ids.cpu().numpy()
        for i in range(n):
            members = self._category_members[int(category_np[i])]
            pos = int(stimulus_np[i])
            if len(members) < 2:
                negatives[i] = pos
                continue
            pos_in_cat = members.index(pos)
            offset = 1 + int(rng.integers(0, len(members) - 1))
            negatives[i] = members[(pos_in_cat + offset) % len(members)]
        return negatives.to(stimulus_indices.device)

    def matching_inputs(
        self,
        batch: Any,
        *,
        enc: EncodedTrials | None = None,
        epoch: int = 0,
    ) -> Stage2MatchInputs:
        """Correct/incorrect bank tensors for the HC-only matching losses.

        Reuses ``enc`` when supplied so the encoder runs at most once per
        batch. HC selection comes from subject labels mapped through flat
        subject slots — never from IDs. With bank features neutralized
        (``no_bank``) there is nothing to match: returns ``None``.
        """
        if not self.bank_features_active:
            return None
        batch.validate()
        flat = flatten_valid_trials(batch)
        if enc is None:
            enc = self.encode_trials(batch, "train")
        device = self.device

        hc_mask = (flat.labels_per_trial.to(device) == 0)
        negative_stimuli = self._sample_negative_stimuli(
            flat.stimulus_indices, flat.category_ids, epoch
        ).to(device)
        eligible = torch.tensor(
            [
                len(self._category_members[int(c)]) >= 2
                for c in flat.category_ids.tolist()
            ],
            dtype=torch.bool,
            device=device,
        )
        hc_match_mask = hc_mask & eligible
        # Placeholder rows for ineligible trials: identical to the positive bank.
        neg_indices = torch.where(eligible, negative_stimuli, flat.stimulus_indices.to(device))

        wrong = self.bank_store.gather_trials(
            stimulus_indices=neg_indices,
            subject_slots=flat.subject_slots,
            subject_bank_ids=batch.bank_split_ids,
            split="train",
            device=device,
        )
        rel_neg = self.relation.forward_with_query(
            enc.q, wrong.mu_trial, wrong.sigma_trial, wrong.count_trial
        )

        token_tensors: dict[str, torch.Tensor | None] = {}
        if self.token_branch is not None:
            n_mu_neg, _, _ = self.token_branch.projections.project_bank(
                wrong.mu_token, wrong.sigma_token, wrong.count_trial
            )
            token_tensors = {
                "Q": enc.Q,
                "N_mu_pos": enc.N_mu,
                "N_mu_neg": n_mu_neg,
                "token_rho": enc.token_rho,
                "token_omega": enc.token_omega,
            }

        return Stage2MatchInputs(
            hc_mask=hc_mask,
            hc_match_mask=hc_match_mask,
            negative_stimulus_indices=negative_stimuli,
            cos_pos=enc.cosine.squeeze(-1),
            cos_neg=rel_neg["cosine"].squeeze(-1),
            comparator_pos=enc.comparator,
            comparator_neg=rel_neg["comparator"],
            rho=enc.rho,
            **token_tensors,
        )

    # ------------------------------------------------------------------ report

    def parameter_report(self) -> dict[str, Any]:
        encoder_report = self.transferred_encoder.parameter_report()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_names": trainable_names,
            "encoder": {
                "total": encoder_report.total,
                "trainable": encoder_report.trainable,
                "frozen": encoder_report.frozen,
                "trainable_names": encoder_report.trainable_names,
            },
        }


# ------------------------------------------------------------------- dry run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 model dry run (no training)")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "stage2" / "base.yaml")
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablation", default=None, help="named ablation overlay to resolve")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="explicit dotted-key override applied after the overlay (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="run one real fold batch and exit")
    return parser


def dry_run(cfg: Stage2Config, device: str, *, ablation_info: dict[str, Any] | None = None) -> dict[str, Any]:
    bank_store = NormativeBankStore(cfg, cfg.fold, device=device)
    model = Stage2Model(cfg, bank_store, device=device)
    report = model.parameter_report()
    print("=== Stage-2 model dry run ===")
    print(
        f"ablation {cfg.ablation} | fold {cfg.fold} | bank mode {model.bank_mode} | "
        f"bank train_mode {bank_store.train_mode}"
    )
    if model.wrong_bank_permutation is not None:
        import json

        print(
            f"wrong-bank permutation (deterministic in seed/fold; Phase 5 persists it "
            f"as wrong_bank_permutation.json in the run audit dir):\n"
            f"  {json.dumps({str(k): v for k, v in sorted(model.wrong_bank_permutation.items())})}"
        )
    if ablation_info is not None:
        print(f"reference configuration: {ablation_info.get('reference')}")
        print(f"changed dotted keys: {ablation_info.get('changed_keys')}")
        print(f"required bank capabilities: {ablation_info.get('required_bank_capabilities')}")
        print(f"config hash: {ablation_info.get('config_hash')}")
    print(
        f"bank root {bank_store.bank_root} | token banks present: {bank_store.has_token_banks}"
    )
    print(
        f"loaded encoder checkpoint: {model.transferred_encoder.checkpoint_path}\n"
        f"  sha256 {model.transferred_encoder.checkpoint_sha256} | "
        f"stage1 run {model.transferred_encoder.stage1_run_id}"
    )
    print(
        f"parameters: total {report['total']} | trainable {report['trainable']} | "
        f"frozen {report['frozen']}"
    )
    print(
        f"encoder parameters: total {report['encoder']['total']} | "
        f"trainable {report['encoder']['trainable']} | frozen {report['encoder']['frozen']}"
    )
    print(f"trainable module names: {report['trainable_names']}")

    train_ds = Stage2SubjectDataset(cfg, cfg.fold, "train", bank_store=bank_store)
    sampler = BalancedSubjectBatchSampler(
        train_ds,
        batch_size=cfg.sampler.subject_batch_size,
        seed=cfg.seed,
        fold=cfg.fold,
        epoch=0,
        balance_groups=cfg.sampler.balance_groups,
        drop_last=cfg.sampler.drop_last,
        shuffle=True,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        collate_fn=collate_subject_samples,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
    )
    batch = next(iter(loader))
    batch = batch.to_device(device)
    print(f"one batch: {batch.n_subjects} subjects | valid trials {batch.n_valid_trials}")

    flat = flatten_valid_trials(batch)
    print(f"flattened inputs: heatmaps {tuple(flat.heatmaps.shape)}")

    from .losses import compute_stage2_losses, generate_subset_masks

    subset_masks = None
    if cfg.subsets.enabled:
        subset_masks = generate_subset_masks(
            trial_mask=batch.trial_mask.to(device),
            category_ids=batch.category_ids.to(device),
            subject_ids=list(batch.subject_ids),
            seed=cfg.seed,
            fold=cfg.fold,
            epoch=0,
            min_fraction=cfg.subsets.min_fraction,
            max_fraction=cfg.subsets.max_fraction,
            train=True,
        )
    result = model(
        batch, "train", subset_masks=subset_masks, debug_token_attention=model.token_branch is not None
    )
    out = result.full
    print("forward shapes:")
    for name, value in [
        ("main_logit", out.main_logit),
        ("auxiliary_logit", out.auxiliary_logit),
        ("subject_embedding", out.subject_embedding),
        ("trial_embeddings", out.trial_embeddings),
        ("trial_mask", out.trial_mask),
        ("query_patch_attention", out.query_patch_attention),
        ("stimulus_attention", out.stimulus_attention),
        ("stimulus_importance", out.stimulus_importance),
        ("stimulus_evidence", out.stimulus_evidence),
        ("stimulus_contribution", out.stimulus_contribution),
        ("semantic_compatibility", out.semantic_compatibility),
        ("normative_deviation", out.normative_deviation),
        ("weighted_normative_deviation", out.weighted_normative_deviation),
        ("semantic_patch_map", out.semantic_patch_map),
    ]:
        print(f"  {name}: {None if value is None else tuple(value.shape)}")
    print(f"  subsets: {sorted(result.subsets)}")
    print(
        f"category present: {out.diagnostics['category_present'].tolist()}"
    )

    enc = model.encode_trials(batch, "train", debug_token_attention=model.token_branch is not None)
    match_inputs = model.matching_inputs(batch, enc=enc, epoch=0)
    anchor_current, anchor_stage1 = model.transferred_encoder.anchor_vectors()
    if cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
        anchor_current = anchor_stage1 = None
    losses = compute_stage2_losses(
        loss_cfg=cfg.loss,
        subsets_cfg=cfg.subsets,
        labels=batch.labels.to(device),
        full=out,
        subsets=result.subsets,
        match_inputs=match_inputs,
        anchor_current=anchor_current,
        anchor_stage1=anchor_stage1,
        epoch=0,
    )
    print("loss components:")
    for name in (
        "total", "cls", "aux", "match", "trialmatch", "bankrank", "tokenmatch",
        "cons", "latent_cons", "prob_cons", "entropy", "anchor",
    ):
        print(f"  {name}: {float(getattr(losses, name).detach())}")
    print(
        f"  hc match trials {losses.n_hc_match_trials} | skipped {losses.n_skipped_match_trials} | "
        f"matched cosine {losses.matched_cosine_mean:.4f} | wrong cosine {losses.wrong_cosine_mean:.4f} | "
        f"bank rank acc {losses.bank_rank_accuracy:.4f}"
    )

    print("gradient audit:")
    losses.total.backward()
    encoder_names = set(report["encoder"]["trainable_names"])
    audit = []
    for name, param in model.named_parameters():
        if param.grad is None:
            audit.append((name, "no-grad"))
        elif not torch.isfinite(param.grad).all() or param.grad.abs().max() == 0.0:
            audit.append((name, "bad-grad"))
        else:
            audit.append((name, "finite-nonzero"))
    for name, status in audit:
        print(f"  {name}: {status}")
    frozen_with_grad = [
        name for name, param in model.transferred_encoder.encoder.named_parameters()
        if param.requires_grad is False and param.grad is not None
    ]
    print(f"frozen encoder params with gradients: {frozen_with_grad or 'none'}")
    print(
        f"bank tensors require grad: "
        f"{any(t.requires_grad for t in [bank_store.mu_trial, bank_store.sigma_trial])}"
    )
    print("dry run complete")
    return {"model": model, "result": result, "losses": losses, "batch": batch}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.ablation:
            from .ablations import resolve_ablation_config

            overrides = {}
            for item in args.override:
                if "=" not in item:
                    raise ConfigError(f"override must be KEY=VALUE, got {item!r}")
                key, value = item.split("=", 1)
                overrides[key.strip()] = value.strip()
            resolved = resolve_ablation_config(
                args.config, args.ablation, overrides=overrides, fold=args.fold
            )
            cfg = resolved.config
            ablation_info = {
                "reference": resolved.spec.reference,
                "changed_keys": sorted(resolved.changed_keys),
                "required_bank_capabilities": list(resolved.spec.required_bank_capabilities),
                "config_hash": resolved.config_hash,
            }
        else:
            cfg = Stage2Config.from_yaml(args.config)
            cfg = Stage2Config.from_dict({**cfg.to_dict(), "fold": args.fold})
            ablation_info = None
        if not args.dry_run:
            print("use --dry-run to run one real fold batch through the model (no training exists yet)")
            return 0
        dry_run(cfg, args.device, ablation_info=ablation_info)
        return 0
    except (ConfigError, Stage2ModelError, TransferredEncoderError, ValueError, OSError) as exc:
        print(f"stage2 model check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Leave-one-cohort-out nested validation: primary model, baselines and ablations.

Outer loop: hold out one entire cohort and train on the rest. The held-out cohort
is invisible during feature screening, hyperparameter choice and threshold setting.
Inner loop: leave-one-training-cohort-out over the training cohorts (falls back to
stratified 5-fold when only one remains); chooses pair count, hyperparameters and threshold.

Models (plan sections 10.5 / 10.6):
  DEG_LogReg / DEG_LASSO / DEG_RF / DEG_XGB   conventional absolute expression
  Module_LogReg                               single-cell module scores only
  DRGpair_LASSO                               genome-wide rank pairs, no single-cell constraint
  scPair_LASSO                                cell-state-constrained rank pairs + LASSO
  DeepPair                                    DeepSets without program labels
  scDRP_DKD                                   full model
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                       # noqa: E402
from scdrp import data as D                             # noqa: E402
from scdrp import metrics as MT                         # noqa: E402
from scdrp import models as MD                          # noqa: E402
from scdrp import screening as SC                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
MODELS = ROOT / "models"

SEED = 20260722
SEEDS = [20260722 + i for i in range(10)]
N_PAIR_GRID = [50, 100, 150, 200, 300]
DEG_GRID = [25, 50, 100]
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# pair features -> tensors
# --------------------------------------------------------------------------- #
def pair_tensors(cds: list[D.CohortData], pair_idx: np.ndarray):
    P, M, mask, y, cid = D.stack(cds, pair_idx)
    return dict(P=torch.tensor(P.T, dtype=torch.float32),
                M=torch.tensor(M.T, dtype=torch.float32),
                mask=torch.tensor(mask.T, dtype=torch.bool),
                y=torch.tensor(y, dtype=torch.float32)), y, cid


def label_tensors(comp: D.Compartment, pair_idx: np.ndarray,
                  gene_map, cell_map, prog_map):
    sub = comp.pairs.iloc[pair_idx]
    to = lambda v: torch.tensor(np.asarray(v), dtype=torch.long)   # noqa: E731
    return dict(
        ga=to([gene_map[g] for g in sub.gene_a]),
        gb=to([gene_map[g] for g in sub.gene_b]),
        ca=to([cell_map[c] for c in sub.cell_a]),
        cb=to([cell_map[c] for c in sub.cell_b]),
        pa=to([prog_map[p] for p in sub.program_a]),
        pb=to([prog_map[p] for p in sub.program_b]),
        cat=to(sub.category.values - 1))


def flat_pair_features(cds: list[D.CohortData], pair_idx: np.ndarray):
    """For LASSO: direction + margin stacked into a dense samples x (2*pairs) matrix."""
    P, M, mask, y, cid = D.stack(cds, pair_idx)
    P = np.where(mask, P, 0.5).T.astype(float)      # unmeasured pairs get the neutral value
    M = np.where(mask, M, 0.0).T.astype(float)
    return np.hstack([P, M]), y, cid


# --------------------------------------------------------------------------- #
# inner loop: choosing the number of pairs
# --------------------------------------------------------------------------- #
def inner_splits(cid: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Inner leave-one-cohort-out with >=2 training cohorts, else stratified 5-fold."""
    uniq = np.unique(cid)
    if len(uniq) >= 2:
        return [(np.flatnonzero(cid != c), np.flatnonzero(cid == c)) for c in uniq]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=min(5, int(np.bincount(y).min())),
                          shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def oof_predict(fit, X: np.ndarray, y: np.ndarray, cid: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions within the training set. The threshold must come from
    these: resubstitution predictions push it into the extreme tail of the fitted probability distribution (earlier iterations here gave thresholds above 0.99 and zero sensitivity)."""
    p = np.full(len(y), np.nan)
    for tr, va in inner_splits(cid, y):
        if len(np.unique(y[tr])) < 2:
            continue
        p[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    if np.isnan(p).any():                     # fall back to a full fit in degenerate cases
        m = fit(X, y)
        p[np.isnan(p)] = m.predict_proba(X)[np.isnan(p), 1]
    return p


def choose_n_pairs(selected: pd.DataFrame, cds, comp, maps) -> int:
    """Choose the number of pairs entering the model by AUROC on the inner CV."""
    best_n, best_auc = min(N_PAIR_GRID[0], len(selected)), -1.0
    for n in N_PAIR_GRID:
        if n > len(selected):
            break
        idx = selected.pair_id.values[:n]
        X, y, cid = flat_pair_features(cds, idx)
        aucs = []
        for tr, va in inner_splits(cid, y):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            m = bl.fit_lasso_cv(X[tr], y[tr])
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_n = float(np.mean(aucs)), n
    del comp, maps
    return best_n


# --------------------------------------------------------------------------- #
# a single outer fold
# --------------------------------------------------------------------------- #
def run_fold(comp: D.Compartment, held: str, programs: dict[str, list[str]],
             out_pred: list, out_pairs: list) -> list[dict]:
    train_ids = [c for c in comp.cohorts if c != held]
    tr_cds = [comp.cohorts[c] for c in train_ids]
    te_cds = [comp.cohorts[held]]
    print(f"\n--- fold: held out {held} | training {train_ids} ---")

    # ---------- 1. pair screening inside the training set ----------
    P, M, mask, ytr, cid = D.stack(tr_cds)
    sel = SC.select_pairs(P, M, mask, ytr, cid, comp.pairs, n_max=max(N_PAIR_GRID),
                          seed=SEED)
    if sel.empty:
        print("  [warn] no gene pair passed screening")
        return []
    # Stable sort: stability values tie heavily, so break ties deterministically on
    # |delta| and pair_id; an unstable sort shifted held-out AUROC by up to 0.017.
    sel = (sel.assign(_absdelta=sel.delta.abs())
           .sort_values(["stability", "_absdelta", "pair_id"],
                        ascending=[False, False, True], kind="mergesort")
           .drop(columns="_absdelta").reset_index(drop=True))
    n_pairs = choose_n_pairs(sel, tr_cds, comp, None)
    keep = sel.head(n_pairs)
    pair_idx = keep.pair_id.values
    print(f"  {len(sel)} pairs passed screening, inner CV chose {n_pairs} "
          f"(category counts {keep.category.value_counts().to_dict()})")
    out_pairs.append(keep.assign(compartment=comp.name, held_out=held))

    gene_map = {g: i for i, g in enumerate(comp.genes)}
    cells = sorted(set(comp.pairs.cell_a) | set(comp.pairs.cell_b))
    progs = sorted(set(comp.pairs.program_a) | set(comp.pairs.program_b))
    cell_map = {c: i for i, c in enumerate(cells)}
    prog_map = {p: i for i, p in enumerate(progs)}
    labels = label_tensors(comp, pair_idx, gene_map, cell_map, prog_map)

    results = []

    def record(name: str, p_tr: np.ndarray, y_tr: np.ndarray,
               p_te: np.ndarray, y_te: np.ndarray, extra: dict | None = None):
        thr = MT.youden_threshold(y_tr, p_tr)
        row = dict(compartment=comp.name, held_out=held, model=name)
        row.update(MT.evaluate(y_te, p_te, thr))
        lo, hi = MT.bootstrap_ci(y_te, p_te, "auroc", seed=SEED)
        row["auroc_ci_low"], row["auroc_ci_high"] = lo, hi
        lo, hi = MT.bootstrap_ci(y_te, p_te, "auprc", seed=SEED)
        row["auprc_ci_low"], row["auprc_ci_high"] = lo, hi
        row["n_pairs"] = n_pairs
        if extra:
            row.update(extra)
        results.append(row)
        out_pred.append(pd.DataFrame(dict(
            compartment=comp.name, held_out=held, model=name,
            sample=te_cds[0].samples, y=y_te, p=p_te)))
        print(f"  {name:16s} AUROC={row['auroc']:.3f} "
              f"[{row['auroc_ci_low']:.2f},{row['auroc_ci_high']:.2f}] "
              f"AUPRC={row['auprc']:.3f} BA={row['balanced_accuracy']:.3f} "
              f"Brier={row['brier']:.3f}")

    # ---------- 2. conventional expression baselines ----------
    genes = comp.universe
    Xtr, ytr_e, cid_e = D.expression_matrix(tr_cds, genes)
    Xte, yte_e, _ = D.expression_matrix(te_cds, genes)

    best_deg, best_auc = DEG_GRID[0], -1.0
    for n_deg in DEG_GRID:
        aucs = []
        for tr, va in inner_splits(cid_e, ytr_e):
            if len(np.unique(ytr_e[va])) < 2:
                continue
            order, _ = bl.select_degs(Xtr[tr], ytr_e[tr], genes, n_deg)
            m = bl.fit_lasso_cv(Xtr[tr][:, order], ytr_e[tr])
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(ytr_e[va],
                                      m.predict_proba(Xtr[va][:, order])[:, 1]))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_deg = float(np.mean(aucs)), n_deg
    order, deg_tab = bl.select_degs(Xtr, ytr_e, genes, best_deg)
    print(f"  inner CV chose {best_deg} differentially expressed genes")

    for name, fit in (("DEG_LogReg", bl.fit_logistic), ("DEG_LASSO", bl.fit_lasso_cv),
                      ("DEG_RF", bl.fit_rf), ("DEG_XGB", bl.fit_xgb)):
        m = fit(Xtr[:, order], ytr_e)
        record(name, oof_predict(fit, Xtr[:, order], ytr_e, cid_e), ytr_e,
               m.predict_proba(Xte[:, order])[:, 1], yte_e,
               dict(n_features=best_deg))

    # ---------- 3. module scores only ----------
    Str = np.vstack([bl.module_scores(c.rank, programs) for c in tr_cds])
    Ste = np.vstack([bl.module_scores(c.rank, programs) for c in te_cds])
    m = bl.fit_logistic(Str, ytr_e)
    record("Module_LogReg", oof_predict(bl.fit_logistic, Str, ytr_e, cid_e), ytr_e,
           m.predict_proba(Ste)[:, 1], yte_e, dict(n_features=Str.shape[1]))

    # ---------- 4. genome-wide rank pairs, no single-cell constraint ----------
    # This is the comparator for the study's central claim and must match
    # scPair-LASSO step for step: the same margin features, the same screening
    # cascade (bootstrap stability selection and redundancy pruning included), and
    # the same top-n_pairs selection by stability. The only difference is that these candidate genes come from genome-wide differential expression within the training set rather than from single-cell programs.
    gw_order, _ = bl.select_degs(Xtr, ytr_e, genes, 200)
    gw_genes = [genes[i] for i in gw_order]
    gw_pairs_idx = [(a, b) for i, a in enumerate(gw_genes)
                    for b in gw_genes[i + 1:]]
    gw_table = pd.DataFrame(dict(
        pair_id=np.arange(len(gw_pairs_idx)),
        gene_a=[a for a, _ in gw_pairs_idx],
        gene_b=[b for _, b in gw_pairs_idx],
        category=np.full(len(gw_pairs_idx), 3, dtype=np.int8)))

    def gw_matrices(cds):
        P_, M_, K_ = [], [], []
        for c in cds:
            a = c.rank.reindex([g for g, _ in gw_pairs_idx]).values
            b = c.rank.reindex([g for _, g in gw_pairs_idx]).values
            d = a - b
            P_.append((d > 0).astype(np.int8))
            M_.append(np.abs(d).astype(np.float32))
            K_.append(np.isfinite(d))
        return np.hstack(P_), np.hstack(M_), np.hstack(K_)

    gwP, gwM, gwK = gw_matrices(tr_cds)
    gw_sel = SC.select_pairs(gwP, gwM, gwK, ytr_e, cid_e, gw_table,
                             n_max=n_pairs, seed=SEED)
    if len(gw_sel) >= 5:
        gid = gw_sel.pair_id.values
        gwPte, gwMte, gwKte = gw_matrices(te_cds)
        Xg_tr = np.hstack([np.where(gwK[gid], gwP[gid], 0.5).T,
                           np.where(gwK[gid], gwM[gid], 0.0).T])
        Xg_te = np.hstack([np.where(gwKte[gid], gwPte[gid], 0.5).T,
                           np.where(gwKte[gid], gwMte[gid], 0.0).T])
        m = bl.fit_lasso_cv(Xg_tr, ytr_e)
        record("DRGpair_LASSO",
               oof_predict(bl.fit_lasso_cv, Xg_tr, ytr_e, cid_e), ytr_e,
               m.predict_proba(Xg_te)[:, 1], yte_e,
               dict(n_features=len(gid)))
    else:
        print("  [warn] DRGpair baseline did not yield enough significant pairs")

    # ---------- 5. cell-state-constrained pairs + LASSO ----------
    Xp_tr, y_tr2, _ = flat_pair_features(tr_cds, pair_idx)
    Xp_te, y_te2, _ = flat_pair_features(te_cds, pair_idx)
    m = bl.fit_lasso_cv(Xp_tr, y_tr2)
    _, _, cid_p = flat_pair_features(tr_cds, pair_idx)
    record("scPair_LASSO", oof_predict(bl.fit_lasso_cv, Xp_tr, y_tr2, cid_p),
           y_tr2, m.predict_proba(Xp_te)[:, 1], y_te2,
           dict(n_features=n_pairs))

    # ---------- 6. DeepSets (no program labels / full model) ----------
    tr_t, y_tr3, cid3 = pair_tensors(tr_cds, pair_idx)
    te_t, y_te3, _ = pair_tensors(te_cds, pair_idx)
    # Deep model: inner stratified 5-fold gives out-of-fold logits, which fit a Platt
    # calibration and lock the threshold; the final model is retrained on all training
    # data for the inner median best epoch count, and the same Platt mapping is applied to its held-out predictions. The held-out cohort never enters any fit.
    from sklearn.model_selection import StratifiedKFold
    pos_w = float((y_tr3 == 0).sum() / max(1, (y_tr3 == 1).sum()))

    for name, use_prog in (("DeepPair", False), ("scDRP_DKD", True)):
        cfg = MD.DeepSetsConfig(n_genes=len(comp.genes), n_cells=len(cells),
                                n_programs=len(progs), use_program=use_prog,
                                pos_weight=pos_w)
        # Each seed splits the training set in half and trains one small model per
        # half. Out-of-fold predictions average one model per seed (len(SEEDS) in
        # total); held-out predictions average one model per seed as well. Both sides
        # therefore average the same number of equally sized models, so the logit scales are comparable and the Platt calibration transfers directly.
        oof_logit = np.zeros((len(y_tr3), len(SEEDS)))
        te_logits, alphas, n_par = [], [], 0
        for si, sd in enumerate(SEEDS):
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=sd)
            fold_models = []
            for tr, va in skf.split(np.zeros(len(y_tr3)), y_tr3):
                sub_tr = {k: v[tr] for k, v in tr_t.items()}
                sub_va = {k: v[va] for k, v in tr_t.items()}
                m_k, _ = MD.train_deepsets(cfg, sub_tr, sub_va, labels,
                                           DEVICE, sd)
                oof_logit[va, si] = MD.predict(m_k, sub_va, labels)[0]
                fold_models.append(m_k)
                n_par = MD.n_parameters(m_k)
            pick = fold_models[si % len(fold_models)]
            lg, _, al = MD.predict(pick, te_t, labels)
            te_logits.append(lg)
            alphas.append(al)
        oof_mean = oof_logit.mean(axis=1)
        te_mean = np.mean(te_logits, axis=0)
        platt = MD.fit_platt(oof_mean, y_tr3)
        record(name, platt(oof_mean), y_tr3, platt(te_mean), y_te3,
               dict(n_features=n_pairs, n_parameters=n_par,
                    n_models=len(te_logits)))
        if use_prog:
            np.save(MODELS / f"attention_{comp.name}_{held}.npy",
                    np.mean(alphas, axis=0))

    del deg_tab
    return results


def main() -> None:
    MET.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    prog_tab = pd.read_csv(ROOT / "data_processed" / "programs" /
                           "programs_raw.tsv", sep="\t")
    programs = {r.program: str(r.genes).split(",") for r in prog_tab.itertuples()}

    all_res, all_pred, all_pairs = [], [], []
    for comp_name in ("GLOM", "TUB"):
        comp = D.load_compartment(comp_name)
        print(f"\n===== compartment {comp_name}: {len(comp.cohorts)} cohorts, "
              f"{comp.pairs.shape[0]} candidate gene pairs =====")
        for cid, cd in comp.cohorts.items():
            print(f"  {cid}: n={len(cd.y)} (DKD {int(cd.y.sum())})")
        for held in comp.cohorts:
            all_res += run_fold(comp, held, programs, all_pred, all_pairs)

    res = pd.DataFrame(all_res)
    res.to_csv(MET / "loco_results.tsv", sep="\t", index=False)
    pd.concat(all_pred, ignore_index=True).to_csv(
        MET / "loco_predictions.tsv.gz", sep="\t", index=False)
    pd.concat(all_pairs, ignore_index=True).to_csv(
        TAB / "T12_selected_pairs.tsv.gz", sep="\t", index=False)

    macro = (res.groupby(["compartment", "model"])
             .agg(macro_auroc=("auroc", "mean"), macro_auprc=("auprc", "mean"),
                  macro_ba=("balanced_accuracy", "mean"),
                  macro_brier=("brier", "mean"),
                  mean_cal_slope=("calibration_slope", "mean"))
             .reset_index().sort_values(["compartment", "macro_auroc"],
                                        ascending=[True, False]))
    macro.to_csv(TAB / "T13_macro_performance.tsv", sep="\t", index=False)
    print("\n===== LOCO macro performance =====")
    print(macro.to_string(index=False))
    (MET / "config.json").write_text(json.dumps(
        dict(seed=SEED, n_pair_grid=N_PAIR_GRID, deg_grid=DEG_GRID), indent=2))


if __name__ == "__main__":
    main()

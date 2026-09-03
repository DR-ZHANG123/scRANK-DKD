#!/usr/bin/env python
"""Apparent gain produced by the construction of the comparator itself.

The primary baseline, DRGpair-LASSO, is matched to scPair-LASSO step for step: the
same margin features, the same screening cascade (bootstrap stability selection and
redundancy pruning), the same number of pairs. The only difference is that its
candidate genes come from genome-wide differential expression rather than from
single-cell programs.

Comparators of this kind in the literature are often not matched: direction features
only, no stability selection, no redundancy pruning, just the top n pairs by a
univariate chi-square within the training set. This script evaluates both
constructions on identical folds and identical candidate genes, quantifying how much
apparent gain the construction alone can manufacture. The scPair-LASSO numbers are
taken from 12 and are not recomputed.

Writes results/tables/T41_comparator_construction.tsv"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                       # noqa: E402
from scdrp import data as D                             # noqa: E402
from scdrp import metrics as MT                         # noqa: E402
from scdrp import screening as SC                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
SEED = 20260722
N_DEG = 200


def gw_matrices(cds, pairs: list[tuple[str, str]]):
    """Direction, margin and availability matrices for genome-wide pairs (pairs x samples)."""
    P_, M_, K_ = [], [], []
    for c in cds:
        a = c.rank.reindex([g for g, _ in pairs]).values
        b = c.rank.reindex([g for _, g in pairs]).values
        d = a - b
        P_.append((d > 0).astype(np.int8))
        M_.append(np.abs(d).astype(np.float32))
        K_.append(np.isfinite(d))
    return np.hstack(P_), np.hstack(M_), np.hstack(K_)


def chi2_rank(P: np.ndarray, keep: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank pairs by their univariate chi-square statistic, with no multiplicity correction."""
    out = np.full(P.shape[0], -np.inf)
    for i in np.where(keep)[0]:
        tab = np.array([[int(((P[i] == v) & (y == c)).sum()) for v in (0, 1)]
                        for c in (0, 1)], dtype=float) + 0.5
        out[i] = stats.chi2_contingency(tab)[0]
    return out


def fit_eval(Xtr, ytr, cidtr, Xte, yte, keep_p: list | None = None) -> dict:
    m = bl.fit_lasso_cv(Xtr, ytr)
    p_tr = np.zeros(len(ytr))
    for tr, va in _splits(cidtr, ytr):
        if len(np.unique(ytr[tr])) < 2:
            continue
        p_tr[va] = bl.fit_lasso_cv(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
    thr = MT.youden_threshold(ytr, p_tr)
    p_te = m.predict_proba(Xte)[:, 1]
    if keep_p is not None:
        keep_p.append(p_te)
    return MT.evaluate(yte, p_te, thr)


def _splits(cid: np.ndarray, y: np.ndarray):
    """Leave-one-training-cohort-out; falls back to stratified 5-fold with a single training cohort."""
    from sklearn.model_selection import StratifiedKFold
    uc = np.unique(cid)
    if len(uc) > 1:
        for c in uc:
            va = np.where(cid == c)[0]
            tr = np.where(cid != c)[0]
            if len(np.unique(y[va])) == 2 and len(np.unique(y[tr])) == 2:
                yield tr, va
        return
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y):
        yield tr, va


def run_fold(comp: D.Compartment, held: str, n_pairs: int,
             out_pred: list) -> list[dict]:
    train_ids = [c for c in comp.cohorts if c != held]
    tr_cds = [comp.cohorts[c] for c in train_ids]
    te_cds = [comp.cohorts[held]]

    genes = comp.universe
    Xtr, ytr, cid = D.expression_matrix(tr_cds, genes)
    _, yte, _ = D.expression_matrix(te_cds, genes)
    order, _ = bl.select_degs(Xtr, ytr, genes, N_DEG)
    gw_genes = [genes[i] for i in order]
    pairs = [(a, b) for i, a in enumerate(gw_genes) for b in gw_genes[i + 1:]]
    table = pd.DataFrame(dict(pair_id=np.arange(len(pairs)),
                              gene_a=[a for a, _ in pairs],
                              gene_b=[b for _, b in pairs],
                              category=np.full(len(pairs), 3, dtype=np.int8)))

    P, M, K = gw_matrices(tr_cds, pairs)
    Pte, Mte, Kte = gw_matrices(te_cds, pairs)
    rows, preds = [], []

    sel = SC.select_pairs(P, M, K, ytr, cid, table, n_max=n_pairs, seed=SEED)
    if len(sel) >= 5:
        g = sel.pair_id.values
        A = np.hstack([np.where(K[g], P[g], 0.5).T, np.where(K[g], M[g], 0.0).T])
        B = np.hstack([np.where(Kte[g], Pte[g], 0.5).T,
                       np.where(Kte[g], Mte[g], 0.0).T])
        rows.append(dict(compartment=comp.name, held_out=held,
                         comparator="matched", n_features=len(g),
                         **fit_eval(A, ytr, cid, B, yte, preds)))

    if len(sel) >= 5:
        g = sel.pair_id.values
        rows.append(dict(compartment=comp.name, held_out=held,
                         comparator="matched_direction_only", n_features=len(g),
                         **fit_eval(np.where(K[g], P[g], 0.5).T.astype(np.float32),
                                    ytr, cid,
                                    np.where(Kte[g], Pte[g], 0.5).T
                                    .astype(np.float32), yte, preds)))

    keep = K.all(axis=1)
    if keep.sum() >= 5:
        chi = chi2_rank(P, keep, ytr)
        top = np.argsort(-chi, kind="mergesort")[:n_pairs]
        rows.append(dict(compartment=comp.name, held_out=held,
                         comparator="unmatched", n_features=len(top),
                         **fit_eval(P[top].T.astype(np.float32), ytr, cid,
                                    np.where(Kte[top], Pte[top], 0.5).T
                                    .astype(np.float32), yte, preds)))
    for r, pv in zip(rows, preds):
        out_pred.append(pd.DataFrame(dict(
            compartment=comp.name, held_out=held, model=r["comparator"],
            sample=te_cds[0].samples, y=yte, p=pv)))
        print(f"  {held:12s} {r['comparator']:22s} AUROC={r['auroc']:.3f}")
    return rows


def main() -> None:
    loco = pd.read_csv(MET / "loco_results.tsv", sep="\t")
    n_by_fold = (loco[loco.model == "scPair_LASSO"]
                 .set_index(["compartment", "held_out"]).n_pairs.to_dict())
    all_rows, all_pred = [], []
    for cname in ("GLOM", "TUB"):
        comp = D.load_compartment(cname)
        print(f"\n===== {cname} =====")
        for held in comp.cohorts:
            all_rows += run_fold(comp, held, int(n_by_fold[(cname, held)]),
                                 all_pred)

    df = pd.DataFrame(all_rows)
    macro = (df.groupby(["compartment", "comparator"]).auroc.mean()
             .unstack("comparator"))
    sc = (loco[loco.model == "scPair_LASSO"].groupby("compartment").auroc.mean())
    macro["scPair_LASSO"] = sc
    macro["delta_vs_matched"] = macro["scPair_LASSO"] - macro["matched"]
    macro["delta_vs_unmatched"] = macro["scPair_LASSO"] - macro["unmatched"]
    macro = macro.reset_index()
    df.to_csv(TAB / "T41_comparator_construction.tsv", sep="\t", index=False)
    macro.to_csv(TAB / "T41b_comparator_macro.tsv", sep="\t", index=False)
    print("\n===== effect of comparator construction (macro AUROC) =====")
    print(macro.to_string(index=False))

    sc_pred = pd.read_csv(MET / "loco_predictions.tsv.gz", sep="\t")
    sc_pred = sc_pred[sc_pred.model == "scPair_LASSO"]
    pool = pd.concat([pd.concat(all_pred, ignore_index=True), sc_pred],
                     ignore_index=True)
    pool["p"] = (pool.groupby(["compartment", "held_out", "model"])["p"]
                 .rank(pct=True))
    drows = []
    for cmp_name, sub in pool.groupby("compartment"):
        wide = sub.pivot_table(index=["held_out", "sample"], columns="model",
                               values="p")
        y = (sub.drop_duplicates(["held_out", "sample"])
             .set_index(["held_out", "sample"])["y"].reindex(wide.index))
        for arm in ("matched", "matched_direction_only", "unmatched"):
            d = MT.paired_bootstrap_delta(y.values, wide["scPair_LASSO"].values,
                                          wide[arm].values)
            drows.append(dict(compartment=cmp_name, comparator=arm,
                              comparator_pooled_auroc=float(MT.evaluate(
                                  y.values, wide[arm].values, 0.5)["auroc"]),
                              **d))
    dd = pd.DataFrame(drows)
    dd.to_csv(TAB / "T41c_comparator_pooled_delta.tsv", sep="\t", index=False)
    print("\n===== pooled paired bootstrap of scPair-LASSO against each comparator =====")
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Whether the screening cascade also costs performance on the constrained pool (post hoc).

12b showed that on the genome-wide pair space, the cascade of BH correction plus
bootstrap stability selection plus redundancy pruning costs about 0.10 AUROC of
external transfer. The immediate inference is that it should be equally harmful on
the framework's own single-cell-constrained pool - and if so, the recommended form
of the framework would drop the cascade.

This script changes exactly one thing on identical folds, the identical candidate
pair space (comp.pairs) and the identical number of pairs per fold: the selection rule.

  sc_full_cascade   the full cascade (equals scPair-LASSO, used as a reproduction check)
  sc_no_cascade     top n pairs by univariate chi-square within the training set,
                    with no multiplicity correction, no stability selection and no
                    redundancy pruning

Both arms use direction and margin features and the same out-of-fold threshold
protocol; the held-out cohort enters no fit.

Writes results/tables/T42_cascade_necessity.tsv
       results/tables/T42b_cascade_pooled_delta.tsv"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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


def fit_eval(Xtr, ytr, cidtr, Xte, yte, keep_p: list) -> dict:
    """Fit the penalized model, set the threshold out of fold, and evaluate on the held-out cohort."""
    m = bl.fit_lasso_cv(Xtr, ytr)
    p_tr = np.zeros(len(ytr))
    for tr, va in _splits(cidtr, ytr):
        if len(np.unique(ytr[tr])) < 2:
            continue
        p_tr[va] = bl.fit_lasso_cv(Xtr[tr], ytr[tr]).predict_proba(Xtr[va])[:, 1]
    p_te = m.predict_proba(Xte)[:, 1]
    keep_p.append(p_te)
    return MT.evaluate(yte, p_te, MT.youden_threshold(ytr, p_tr))


def chi2_stat(P: np.ndarray, mask: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised univariate chi-square statistic for every pair."""
    a, b, c, d = SC._counts(P, mask, y)
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * (a * d - b * c) ** 2.0
        den = (a + b) * (c + d) * (a + c) * (b + d)
        return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def features(P, M, mask, idx):
    return np.hstack([np.where(mask[idx], P[idx], 0.5).T,
                      np.where(mask[idx], M[idx], 0.0).T])


def run_fold(comp: D.Compartment, held: str, n_pairs: int,
             out_pred: list) -> list[dict]:
    tr_cds = [c for k, c in comp.cohorts.items() if k != held]
    te_cds = [comp.cohorts[held]]
    P, M, mask, ytr, cid = D.stack(tr_cds)
    Pte, Mte, mte, yte, _ = D.stack(te_cds)
    rows, preds = [], []

    sel = SC.select_pairs(P, M, mask, ytr, cid, comp.pairs, n_max=n_pairs,
                          seed=SEED)
    if len(sel) >= 5:
        g = sel.pair_id.values[:n_pairs]
        rows.append(dict(compartment=comp.name, held_out=held,
                         arm="sc_full_cascade", n_features=len(g),
                         **fit_eval(features(P, M, mask, g), ytr, cid,
                                    features(Pte, Mte, mte, g), yte, preds)))

    avail = mask.all(axis=1)
    stat = np.where(avail, chi2_stat(P, mask, ytr), -np.inf)
    top = np.argsort(-stat, kind="mergesort")[:n_pairs]
    rows.append(dict(compartment=comp.name, held_out=held,
                     arm="sc_no_cascade", n_features=len(top),
                     **fit_eval(features(P, M, mask, top), ytr, cid,
                                features(Pte, Mte, mte, top), yte, preds)))

    for r, pv in zip(rows, preds):
        out_pred.append(pd.DataFrame(dict(
            compartment=comp.name, held_out=held, model=r["arm"],
            sample=te_cds[0].samples, y=yte, p=pv)))
        print(f"  {held:14s} {r['arm']:16s} AUROC={r['auroc']:.3f}")
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
    macro = df.groupby(["compartment", "arm"]).auroc.mean().unstack("arm")
    macro["scPair_LASSO_primary"] = (loco[loco.model == "scPair_LASSO"]
                                     .groupby("compartment").auroc.mean())
    macro = macro.reset_index()
    df.to_csv(TAB / "T42_cascade_necessity.tsv", sep="\t", index=False)
    print("\n===== effect of the cascade on the constrained candidate pool (macro AUROC) =====")
    print(macro.to_string(index=False))

    pool = pd.concat(all_pred, ignore_index=True)
    pool["p"] = (pool.groupby(["compartment", "held_out", "model"])["p"]
                 .rank(pct=True))
    drows = []
    for cname, sub in pool.groupby("compartment"):
        wide = sub.pivot_table(index=["held_out", "sample"], columns="model",
                               values="p")
        y = (sub.drop_duplicates(["held_out", "sample"])
             .set_index(["held_out", "sample"])["y"].reindex(wide.index))
        d = MT.paired_bootstrap_delta(y.values, wide["sc_no_cascade"].values,
                                      wide["sc_full_cascade"].values)
        drows.append(dict(compartment=cname, contrast="no_cascade - full_cascade",
                          **d))
    dd = pd.DataFrame(drows)
    dd.to_csv(TAB / "T42b_cascade_pooled_delta.tsv", sep="\t", index=False)
    print("\n===== pooled paired bootstrap with the cascade removed =====")
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()

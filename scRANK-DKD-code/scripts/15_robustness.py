#!/usr/bin/env python
"""Negative controls and robustness analyses (plan section 13).

13.1 random gene sets - draw equally many genes from matched expression deciles and
     build the same number of pairs;
13.2 shuffled program labels - permute cell and program labels to test whether the
     labels contribute at all;
13.3 label permutation - shuffle DKD status within the training set; performance
     should return to chance;
13.4 sensitivity to pair count - 20 / 50 / 100 / 200 / 300;
13.5 platform sensitivity - train on one cohort and test on an independent one, and
     drop the largest cohort;
13.6 composition sensitivity - adjust for proportions estimated by reference-based NNLS
     deconvolution."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                        # noqa: E402
from scdrp import data as D                              # noqa: E402
from scdrp import screening as SC                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data_processed" / "bulk"
PAIR = ROOT / "data_processed" / "pair_matrix"
PROC = ROOT / "data_processed" / "scrna"
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"

SEED = 20260722
N_NULL = 100
N_PERM = 100
PAIR_GRID = [20, 50, 100, 200, 300]


def pair_features(P, M, mask, idx):
    return np.hstack([np.where(mask[idx], P[idx], 0.5).T,
                      np.where(mask[idx], M[idx], 0.0).T])


def fit_and_score(Ptr, Mtr, mtr, ytr, cidtr, pairs, idx_te, comp, held,
                  n_max=100, seed=SEED):
    sel = SC.select_pairs(Ptr, Mtr, mtr, ytr, cidtr, pairs, n_max=n_max, seed=seed)
    if sel.empty:
        return None, None
    pid = sel.pair_id.values
    model = bl.fit_lasso_cv(pair_features(Ptr, Mtr, mtr, pid), ytr)
    del comp, held, idx_te
    return pid, model


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def random_gene_null(comp_name: str, rng) -> pd.DataFrame:
    comp = D.load_compartment(comp_name)
    cand = set(comp.genes)
    ref = list(comp.cohorts.values())[0]
    mean_expr = ref.expr.mean(axis=1)
    bins = pd.qcut(mean_expr, 10, labels=False, duplicates="drop")
    cand_bins = bins.reindex(list(cand)).dropna()
    quota = cand_bins.value_counts().to_dict()
    pool = {b: [g for g in bins[bins == b].index if g not in cand]
            for b in quota}

    rows = []
    for it in range(N_NULL):
        genes = []
        for b, k in quota.items():
            avail = pool.get(b, [])
            if avail:
                genes += list(rng.choice(avail, size=min(k, len(avail)),
                                         replace=False))
        genes = sorted(set(genes))
        if len(genes) < 50:
            continue
        aucs = []
        for held in comp.cohorts:
            tr = [c for k, c in comp.cohorts.items() if k != held]
            te = comp.cohorts[held]
            Rtr = [c.rank.reindex(genes) for c in tr]
            gp = [(genes[i], genes[j])
                  for i in range(0, len(genes), 4)
                  for j in range(i + 1, min(i + 5, len(genes)))]
            Ptr, mtr = [], []
            for r in Rtr:
                a = r.loc[[g for g, _ in gp]].values
                b_ = r.loc[[g for _, g in gp]].values
                d = a - b_
                Ptr.append((d > 0).astype(np.int8))
                mtr.append(np.isfinite(d))
            Ptr, mtr = np.hstack(Ptr), np.hstack(mtr)
            ytr = np.concatenate([c.y for c in tr])
            cidtr = np.concatenate([np.full(len(c.y), i) for i, c in enumerate(tr)])
            stats = SC.screen_pairs(Ptr, np.ones_like(Ptr, float), mtr,
                                    ytr, cidtr)
            keep = np.flatnonzero(stats.passed.values)[:100]
            if keep.size < 5:
                continue
            Xtr = np.where(mtr[keep], Ptr[keep], 0.5).T
            m = bl.fit_lasso_cv(Xtr, ytr)
            rte = te.rank.reindex(genes)
            a = rte.loc[[gp[i][0] for i in keep]].values
            b_ = rte.loc[[gp[i][1] for i in keep]].values
            d = a - b_
            Xte = np.where(np.isfinite(d), (d > 0).astype(float), 0.5).T
            aucs.append(roc_auc_score(te.y, m.predict_proba(Xte)[:, 1]))
        if aucs:
            rows.append(dict(compartment=comp_name, iteration=it,
                             macro_auroc=float(np.mean(aucs))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def label_permutation(comp_name: str, rng) -> pd.DataFrame:
    comp = D.load_compartment(comp_name)
    rows = []
    for it in range(N_PERM):
        aucs = []
        for held in comp.cohorts:
            tr = [c for k, c in comp.cohorts.items() if k != held]
            te = comp.cohorts[held]
            P, M, mask, y, cid = D.stack(tr)
            y_perm = np.concatenate([rng.permutation(c.y) for c in tr])
            pid, model = fit_and_score(P, M, mask, y_perm, cid, comp.pairs,
                                       None, comp, held, n_max=100,
                                       seed=SEED + it)
            if pid is None:
                continue
            Xte = pair_features(te.P, te.M, te.mask, pid)
            aucs.append(roc_auc_score(te.y, model.predict_proba(Xte)[:, 1]))
        if aucs:
            rows.append(dict(compartment=comp_name, iteration=it,
                             macro_auroc=float(np.mean(aucs))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def sensitivity(comp_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = D.load_compartment(comp_name)
    n_rows, plat_rows = [], []
    for held in comp.cohorts:
        tr = [c for k, c in comp.cohorts.items() if k != held]
        te = comp.cohorts[held]
        P, M, mask, y, cid = D.stack(tr)
        sel = SC.select_pairs(P, M, mask, y, cid, comp.pairs, n_max=max(PAIR_GRID))
        for n in PAIR_GRID:
            if n > len(sel):
                continue
            pid = sel.pair_id.values[:n]
            m = bl.fit_lasso_cv(pair_features(P, M, mask, pid), y)
            auc = roc_auc_score(te.y, m.predict_proba(
                pair_features(te.P, te.M, te.mask, pid))[:, 1])
            n_rows.append(dict(compartment=comp_name, held_out=held,
                               n_pairs=n, auroc=float(auc)))

        for one in tr:
            P1, M1, k1, y1, c1 = D.stack([one])
            s1 = SC.select_pairs(P1, M1, k1, y1, c1, comp.pairs, n_max=100)
            if s1.empty:
                continue
            pid = s1.pair_id.values
            m = bl.fit_lasso_cv(pair_features(P1, M1, k1, pid), y1)
            auc = roc_auc_score(te.y, m.predict_proba(
                pair_features(te.P, te.M, te.mask, pid))[:, 1])
            plat_rows.append(dict(compartment=comp_name, train=one.cohort,
                                  test=held, n_train=len(y1),
                                  auroc=float(auc)))
    return pd.DataFrame(n_rows), pd.DataFrame(plat_rows)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def deconvolve(comp_name: str) -> pd.DataFrame:
    """Non-negative least squares deconvolution of bulk samples against the single-cell reference."""
    ad = sc.read_h5ad(PROC / "GSE131882_annotated.h5ad")
    expr = ad.raw.to_adata() if ad.raw is not None else ad
    types = [t for t in ad.obs.cell_type.unique()
             if t not in ("Unassigned", "LowQuality")]
    ref = {}
    for t in types:
        sub = expr[ad.obs.cell_type.values == t].X
        ref[t] = np.asarray(sub.mean(axis=0)).ravel()
    ref = pd.DataFrame(ref, index=expr.var_names)

    comp = D.load_compartment(comp_name)
    rows = []
    for cid, cd in comp.cohorts.items():
        genes = [g for g in ref.index if g in cd.expr.index]
        B = ref.loc[genes].values
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-9)
        Y = cd.expr.loc[genes].values
        for j, s in enumerate(cd.samples):
            w, _ = nnls(B, Y[:, j])
            w = w / (w.sum() + 1e-9)
            rows.append(dict(compartment=comp_name, cohort=cid, sample=s,
                             y=int(cd.y[j]), **dict(zip(ref.columns, w))))
    return pd.DataFrame(rows)


def composition_adjustment(prop: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Regress outcome on the score plus estimated cell proportions, to test whether the score is a proxy for composition."""
    import statsmodels.api as sm
    rows = []
    for comp_name, sub in prop.groupby("compartment"):
        p = pred[(pred.compartment == comp_name) &
                 (pred.model == "scPair_LASSO")]
        p = p.assign(score_rank=p.groupby("held_out")["p"].rank(pct=True))
        m = sub.merge(p[["sample", "score_rank"]], on="sample", how="inner")
        if len(m) < 20:
            continue
        cells = [c for c in prop.columns
                 if c not in ("compartment", "cohort", "sample", "y")]
        cells = [c for c in cells if m[c].std() > 1e-6]
        for label, X in (("score_only", m[["score_rank"]]),
                         ("score_plus_composition", m[["score_rank"] + cells])):
            Xd = sm.add_constant(X.astype(float))
            try:
                res = sm.Logit(m.y.values, Xd).fit(disp=0, maxiter=200)
                rows.append(dict(compartment=comp_name, model=label,
                                 n=len(m),
                                 score_coef=float(res.params["score_rank"]),
                                 score_p=float(res.pvalues["score_rank"]),
                                 pseudo_r2=float(res.prsquared)))
            except Exception as exc:                       # noqa: BLE001
                rows.append(dict(compartment=comp_name, model=label, n=len(m),
                                 score_coef=np.nan, score_p=np.nan,
                                 pseudo_r2=np.nan, note=str(exc)[:60]))
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    obs = pd.read_csv(TAB / "T13_macro_performance.tsv", sep="\t")
    real = obs[obs.model == "scPair_LASSO"].set_index("compartment").macro_auroc

    nulls, perms, ns, plats = [], [], [], []
    for comp_name in ("GLOM", "TUB"):
        print(f"\n===== {comp_name} =====")
        n_tab, p_tab = sensitivity(comp_name)
        ns.append(n_tab)
        plats.append(p_tab)
        print("sensitivity to the number of pairs:")
        print(n_tab.groupby("n_pairs").auroc.mean().round(3).to_string())

        perm = label_permutation(comp_name, rng)
        perms.append(perm)
        r = float(real[comp_name])
        pval = (perm.macro_auroc >= r).mean() if len(perm) else np.nan
        print(f"label permutation: null median AUROC={perm.macro_auroc.median():.3f}, "
              f"observed={r:.3f}, empirical P={pval:.4f}")

        null = random_gene_null(comp_name, rng)
        nulls.append(null)
        if len(null):
            pv = (null.macro_auroc >= r).mean()
            print(f"random gene sets: null median AUROC={null.macro_auroc.median():.3f}, "
                  f"observed={r:.3f}, empirical P={pv:.4f}")

    pd.concat(ns).to_csv(TAB / "T22_pair_count_sensitivity.tsv", sep="\t",
                         index=False)
    pd.concat(plats).to_csv(TAB / "T23_platform_sensitivity.tsv", sep="\t",
                            index=False)
    pd.concat(perms).to_csv(MET / "label_permutation.tsv.gz", sep="\t",
                            index=False)
    pd.concat(nulls).to_csv(MET / "random_gene_null.tsv.gz", sep="\t",
                            index=False)

    prop = pd.concat([deconvolve(c) for c in ("GLOM", "TUB")], ignore_index=True)
    prop.to_csv(TAB / "T24_deconvolution.tsv.gz", sep="\t", index=False)
    pred = pd.read_csv(MET / "loco_predictions.tsv.gz", sep="\t")
    adj = composition_adjustment(prop, pred)
    adj.to_csv(TAB / "T25_composition_adjustment.tsv", sep="\t", index=False)
    print("\n===== adjustment for cell proportions =====")
    print(adj.to_string(index=False))

    print("\n===== platform sensitivity (train on one cohort, test on an independent one) =====")
    print(pd.concat(plats).to_string(index=False))


if __name__ == "__main__":
    main()

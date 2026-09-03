#!/usr/bin/env python
"""Unbiased recomputation of the negative controls, with empirical P values.

The first version of 15_robustness.py discarded an iteration whenever a fold produced
no gene pair - but producing no pair is exactly the expected outcome under the null,
so discarding those iterations raises the null distribution (only 25 of 100 label
permutations left a result in tubulointerstitium). Here such folds are scored at
AUROC = 0.5 (no valid ranking possible), the same treatment is applied to the random
gene-set null, and empirical P values are reported."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                        # noqa: E402
from scdrp import data as D                              # noqa: E402
from scdrp import screening as SC                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"

SEED = 20260722
N_PERM = 100
N_GENE_NULL = 100
CHANCE = 0.5


def pair_features(P, M, mask, idx):
    return np.hstack([np.where(mask[idx], P[idx], 0.5).T,
                      np.where(mask[idx], M[idx], 0.0).T])


def label_permutation(comp_name: str, rng) -> pd.DataFrame:
    comp = D.load_compartment(comp_name)
    rows = []
    for it in range(N_PERM):
        aucs, n_failed = [], 0
        for held in comp.cohorts:
            tr = [c for k, c in comp.cohorts.items() if k != held]
            te = comp.cohorts[held]
            P, M, mask, y, cid = D.stack(tr)
            y_perm = np.concatenate([rng.permutation(c.y) for c in tr])
            sel = SC.select_pairs(P, M, mask, y_perm, cid, comp.pairs,
                                  n_max=100, seed=SEED + it)
            if sel.empty:
                aucs.append(CHANCE)
                n_failed += 1
                continue
            pid = sel.pair_id.values
            model = bl.fit_lasso_cv(pair_features(P, M, mask, pid), y_perm)
            Xte = pair_features(te.P, te.M, te.mask, pid)
            aucs.append(float(roc_auc_score(te.y,
                                            model.predict_proba(Xte)[:, 1])))
        rows.append(dict(compartment=comp_name, iteration=it,
                         macro_auroc=float(np.mean(aucs)),
                         folds_without_pairs=n_failed))
    return pd.DataFrame(rows)


def random_gene_null(comp_name: str, rng) -> pd.DataFrame:
    """Random gene sets matched on expression decile, passed through the identical pipeline.
    
    The same pair search space, the same margins, the same screening cascade and stability
    selection, and the same estimator. A null denied the full search space or the stability
    machinery is markedly easier to beat and yields optimistic P values."""
    comp = D.load_compartment(comp_name)
    cand = set(comp.genes)
    ref = list(comp.cohorts.values())[0]
    bins = pd.qcut(ref.expr.mean(axis=1), 10, labels=False, duplicates="drop")
    quota = bins.reindex(list(cand)).dropna().value_counts().to_dict()
    pool = {b: [g for g in bins[bins == b].index if g not in cand]
            for b in quota}

    rows = []
    for it in range(N_GENE_NULL):
        genes = []
        for b, k in quota.items():
            avail = pool.get(b, [])
            if avail:
                genes += list(rng.choice(avail, size=min(k, len(avail)),
                                         replace=False))
        genes = sorted(set(genes))
        if len(genes) < 50:
            continue
        idx_i, idx_j = np.triu_indices(len(genes), k=1)
        table = pd.DataFrame(dict(
            pair_id=np.arange(len(idx_i)),
            gene_a=[genes[i] for i in idx_i],
            gene_b=[genes[j] for j in idx_j],
            category=np.full(len(idx_i), 3, dtype=np.int8)))

        def mats(cds, gi=idx_i, gj=idx_j, gs=genes):
            P_, M_, K_ = [], [], []
            for c in cds:
                R = c.rank.reindex(gs).values
                d = R[gi] - R[gj]
                P_.append((d > 0).astype(np.int8))
                M_.append(np.abs(d).astype(np.float32))
                K_.append(np.isfinite(d))
            return np.hstack(P_), np.hstack(M_), np.hstack(K_)

        aucs, n_failed = [], 0
        for held in comp.cohorts:
            tr = [c for k, c in comp.cohorts.items() if k != held]
            te = [comp.cohorts[held]]
            P, M, K, = mats(tr)
            y = np.concatenate([c.y for c in tr])
            cid = np.concatenate([np.full(len(c.y), i)
                                  for i, c in enumerate(tr)])
            sel = SC.select_pairs(P, M, K, y, cid, table, n_max=100,
                                  seed=SEED + it)
            if len(sel) < 5:
                aucs.append(CHANCE)
                n_failed += 1
                continue
            pid = sel.pair_id.values
            Pt, Mt, Kt = mats(te)
            Xtr = np.hstack([np.where(K[pid], P[pid], 0.5).T,
                             np.where(K[pid], M[pid], 0.0).T])
            Xte = np.hstack([np.where(Kt[pid], Pt[pid], 0.5).T,
                             np.where(Kt[pid], Mt[pid], 0.0).T])
            m = bl.fit_lasso_cv(Xtr, y)
            aucs.append(float(roc_auc_score(te[0].y,
                                            m.predict_proba(Xte)[:, 1])))
        rows.append(dict(compartment=comp_name, iteration=it,
                         macro_auroc=float(np.mean(aucs)),
                         folds_without_pairs=n_failed))
    return pd.DataFrame(rows)


def empirical_p(null: np.ndarray, observed: float) -> float:
    """One-sided empirical P value of the observed value against a null distribution."""
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    macro = pd.read_csv(TAB / "T13_macro_performance.tsv", sep="\t")

    perm = pd.concat([label_permutation(c, rng) for c in ("GLOM", "TUB")],
                     ignore_index=True)
    perm.to_csv(MET / "label_permutation.tsv.gz", sep="\t", index=False)

    gene_null = pd.concat([random_gene_null(c, rng) for c in ("GLOM", "TUB")],
                          ignore_index=True)
    gene_null.to_csv(MET / "random_gene_null.tsv.gz", sep="\t", index=False)

    rows = []
    for comp in ("GLOM", "TUB"):
        for model in ("scPair_LASSO", "scDRP_DKD"):
            obs = float(macro[(macro.compartment == comp) &
                              (macro.model == model)].macro_auroc.iloc[0])
            for name, df in (("label_permutation", perm),
                             ("random_gene_sets", gene_null)):
                v = df[df.compartment == comp].macro_auroc.values
                rows.append(dict(
                    compartment=comp, model=model, null=name,
                    observed_macro_auroc=round(obs, 4), n_null=len(v),
                    null_median=round(float(np.median(v)), 4),
                    null_p95=round(float(np.percentile(v, 95)), 4),
                    empirical_p=round(empirical_p(v, obs), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T26_negative_controls.tsv", sep="\t", index=False)
    print(out.to_string(index=False))
    print("\nfraction of folds in which label permutation found no gene pair:")
    print((perm.groupby("compartment").folds_without_pairs.mean() / 3)
          .round(3).to_string())


if __name__ == "__main__":
    main()

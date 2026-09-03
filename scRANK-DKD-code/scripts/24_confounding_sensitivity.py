#!/usr/bin/env python
"""Confounding and robustness sensitivity analyses.

1. Rerun after excluding the cohort whose batch is perfectly collinear with diagnosis
   (TUB_ERCB1, Cramer's V = 1.00);
2. Cohort-level random-effects meta-analysis (DerSimonian-Laird) in place of a simple
   macro average, giving a pooled AUROC, 95 percent CI, heterogeneity I^2 and a
   prediction interval - the analysis should pool cohorts, not only patients;
3. Leave-one-platform-out: each LOCO cohort here occupies its own platform, so per-fold
   AUROC is reported by held-out platform;
4. Control-provenance sensitivity: in tubulointerstitium, rerun using only controls
   identifiable as living donor biopsies, to test whether the result is driven by how
   the control tissue was obtained.

Only scPair-LASSO and its like-for-like baseline DRGpair-LASSO are assessed; the deep
model is a supplementary analysis and is not included here. Writes T37, T38 and Fig12."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import baselines as bl                          # noqa: E402
from scdrp import data as D                                # noqa: E402
from scdrp import figstyle as FS                           # noqa: E402
from scdrp import screening as SC                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"
BULK = ROOT / "data_processed" / "bulk"
META = ROOT / "data_raw" / "metadata"
FIG = ROOT / "figures"

FS.apply_style()
SEED = 20260722
N_PAIRS = 100
COMPS = {"GLOM": "Glomerulus", "TUB": "Tubulointerstitium"}


def flat(P, M, mask, idx):
    return np.hstack([np.where(mask[idx], P[idx], 0.5).T,
                      np.where(mask[idx], M[idx], 0.0).T])


def hanley_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley and McNeil standard error of an AUROC."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) +
         (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(v, 1e-6)))


def loco_per_fold(comp: D.Compartment, model: str) -> pd.DataFrame:
    """Per-fold AUROC for the two models, read from the LOCO results."""
    rows = []
    for held in comp.cohorts:
        tr = [c for k, c in comp.cohorts.items() if k != held]
        te = comp.cohorts[held]
        P, M, mask, y, cid = D.stack(tr)
        if model == "scPair":
            sel = SC.select_pairs(P, M, mask, y, cid, comp.pairs,
                                  n_max=N_PAIRS, seed=SEED)
            if sel.empty:
                continue
            pid = sel.pair_id.values
            m = bl.fit_lasso_cv(flat(P, M, mask, pid), y)
            Pte, Mte, mte, yte, _ = D.stack([te])
            p = m.predict_proba(flat(Pte, Mte, mte, pid))[:, 1]
        else:                                   # DRGpair like-for-like
            genes = comp.universe
            Xtr, ytr_e, cid_e = D.expression_matrix(tr, genes)
            order, _ = bl.select_degs(Xtr, ytr_e, genes, 200)
            gw = [genes[i] for i in order]
            gp = [(a, b) for i, a in enumerate(gw) for b in gw[i + 1:]]
            tab = pd.DataFrame(dict(pair_id=np.arange(len(gp)),
                                    gene_a=[a for a, _ in gp],
                                    gene_b=[b for _, b in gp],
                                    category=np.full(len(gp), 3, np.int8)))

            def mats(cds):
                Ps, Ms, Ks = [], [], []
                for c in cds:
                    a = c.rank.reindex([g for g, _ in gp]).values
                    b = c.rank.reindex([g for _, g in gp]).values
                    d = a - b
                    Ps.append((d > 0).astype(np.int8))
                    Ms.append(np.abs(d).astype(np.float32))
                    Ks.append(np.isfinite(d))
                return np.hstack(Ps), np.hstack(Ms), np.hstack(Ks)
            gP, gM, gK = mats(tr)
            sel = SC.select_pairs(gP, gM, gK, ytr_e, cid_e, tab,
                                  n_max=N_PAIRS, seed=SEED)
            if sel.empty:
                continue
            pid = sel.pair_id.values
            m = bl.fit_lasso_cv(flat(gP, gM, gK, pid), ytr_e)
            tP, tM, tK = mats([te])
            p = m.predict_proba(flat(tP, tM, tK, pid))[:, 1]
        auc = float(roc_auc_score(te.y, p))
        npos, nneg = int(te.y.sum()), int((te.y == 0).sum())
        rows.append(dict(held_out=held, auroc=auc, n=len(te.y),
                         n_pos=npos, se=hanley_se(auc, npos, nneg)))
    return pd.DataFrame(rows)


def dl_meta(auc: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling on the logit scale.
    
    Returns the pooled estimate, its confidence interval, I^2 and a prediction interval
    for a new cohort. With few cohorts the prediction interval is wide, and that is the
    honest summary of how far the estimate extrapolates."""
    auc, se = np.asarray(auc, float), np.asarray(se, float)
    ok = np.isfinite(auc) & np.isfinite(se) & (se > 0)
    auc, se = auc[ok], se[ok]
    k = len(auc)
    if k == 0:
        return {}
    eps = 1e-3
    a = np.clip(auc, eps, 1 - eps)
    y = np.log(a / (1 - a))                       # logit
    sl = se / (a * (1 - a))                        # delta-method SE on logit
    w = 1 / sl ** 2
    fixed = (w * y).sum() / w.sum()
    Q = (w * (y - fixed) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (sl ** 2 + tau2)
    mu = (wr * y).sum() / wr.sum()
    se_pool = np.sqrt(1 / wr.sum())
    i2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    from scipy.stats import t as tdist
    pi_mult = tdist.ppf(0.975, k - 2) if k > 2 else 1.96
    pi = pi_mult * np.sqrt(tau2 + se_pool ** 2)
    expit = lambda z: 1 / (1 + np.exp(-z))        # noqa: E731
    return dict(k=k, pooled=expit(mu),
                ci_low=expit(mu - 1.96 * se_pool),
                ci_high=expit(mu + 1.96 * se_pool),
                pi_low=expit(mu - pi), pi_high=expit(mu + pi),
                tau2=tau2, i2=i2)


def main() -> None:
    summary, perfold = [], []

    scenarios = {
        "all_cohorts": {"GLOM": None, "TUB": None},
        "exclude_batch_confounded": {"GLOM": None,
                                     "TUB": ["TUB_GSE30529", "TUB_ERCB2"]},
    }

    for scen, comps in scenarios.items():
        for comp_name in ("GLOM", "TUB"):
            cohorts = comps[comp_name]
            comp = D.load_compartment(comp_name, cohorts=cohorts)
            for model in ("scPair", "DRGpair"):
                pf = loco_per_fold(comp, model)
                pf["scenario"] = scen
                pf["compartment"] = comp_name
                pf["model"] = model
                perfold.append(pf)
                meta = dl_meta(pf.auroc.values, pf.se.values)
                summary.append(dict(scenario=scen, compartment=comp_name,
                                    model=model, n_cohorts=meta.get("k"),
                                    macro_auroc=round(float(pf.auroc.mean()), 3),
                                    meta_auroc=round(meta.get("pooled", np.nan), 3),
                                    ci_low=round(meta.get("ci_low", np.nan), 3),
                                    ci_high=round(meta.get("ci_high", np.nan), 3),
                                    pi_low=round(meta.get("pi_low", np.nan), 3),
                                    pi_high=round(meta.get("pi_high", np.nan), 3),
                                    i2=round(meta.get("i2", np.nan), 1)))

    summ = pd.DataFrame(summary)
    summ.to_csv(TAB / "T37_confounding_sensitivity.tsv", sep="\t", index=False)
    pf_all = pd.concat(perfold, ignore_index=True)
    pf_all.to_csv(TAB / "T38_perfold_by_scenario.tsv", sep="\t", index=False)

    print("===== cohort-level random-effects meta-analysis =====")
    print(summ.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(bottom=0.16, wspace=0.42, left=0.13, right=0.97)
    for ci, comp_name in enumerate(COMPS):
        ax = axes[ci]
        FS.panel(ax, "ab"[ci])
        rows = summ[summ.compartment == comp_name]
        labels, y = [], 0
        yt, ytl = [], []
        for scen in ("all_cohorts", "exclude_batch_confounded"):
            for model, col in (("scPair", "#C44E52"), ("DRGpair", "#8C8C8C")):
                r = rows[(rows.scenario == scen) & (rows.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.meta_auroc, y,
                            xerr=[[r.meta_auroc - r.ci_low],
                                  [r.ci_high - r.meta_auroc]],
                            fmt="o", color=col, markersize=4, capsize=2, lw=1.1)
                ax.plot([r.pi_low, r.pi_high], [y, y], color=col, lw=0.7,
                        alpha=0.5)
                yt.append(y)
                ytl.append(f"{'scPair' if model=='scPair' else 'DRGpair'}\n"
                           f"{'all' if scen=='all_cohorts' else 'excl. batch'}")
                y += 1
            y += 0.4
        ax.axvline(0.5, color="#999999", ls="--", lw=0.8)
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl, fontsize=5.6)
        ax.set_xlabel("Meta AUROC (95% CI)", fontweight="bold")
        ax.set_title(COMPS[comp_name], fontsize=7, pad=3)
        ax.set_xlim(0.4, 1.02)
    FS.save(fig, FIG, "Fig12_confounding_sensitivity")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""[EXPLORATORY - supports no claim in the manuscript] Rebuild the full deconvolution
basis and recheck the composition-adjusted regression.

The first basis came only from snRNA (GSE131882), lumped every immune cell into one
BCELL class (glomerular deconvolution then returned 33 percent B cells) and lacked
endothelial, macrophage and pericyte subtypes. The two datasets are complementary,
so both are used:
  - vascular, immune, stromal and most epithelial types: GSE209781 (scRNA, fully resolved);
  - the nucleus-specific epithelial types scRNA loses, PODO / PEC / DCT: GSE131882 (snRNA).
Cell-type means are taken over the shared highly variable genes, quantile-aligned
column-wise onto a comparable scale, combined into one signature matrix, and used
for NNLS deconvolution.

T25 is then rechecked: DKD status against the scPair score plus estimated cell
proportions, testing whether the score is only a proxy for composition.
Writes T24b and T25b."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from scdrp import data as D                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCP = ROOT / "data_processed" / "scrna"
BULK = ROOT / "data_processed" / "bulk"
TAB = ROOT / "results" / "tables"
MET = ROOT / "results" / "metrics"

DROP_CT = {"LowQuality", "Unassigned"}
FROM_SN = {"PODO", "PEC", "DCT"}
N_HVG = 2000


def mean_profiles(ds: str) -> tuple[pd.DataFrame, list[str]]:
    ad = sc.read_h5ad(SCP / f"{ds}_annotated.h5ad")
    expr = ad.raw.to_adata() if ad.raw is not None else ad
    keep = ~ad.obs.cell_type.isin(DROP_CT).values
    cts = sorted(pd.unique(ad.obs.cell_type[keep]))
    prof = {}
    for ct in cts:
        m = keep & (ad.obs.cell_type.values == ct)
        prof[ct] = np.asarray(expr[m].X.mean(axis=0)).ravel()
    return pd.DataFrame(prof, index=expr.var_names), cts


def build_signature() -> pd.DataFrame:
    """Combine the two datasets into one signature matrix on a comparable scale."""
    sc_prof, sc_cts = mean_profiles("GSE209781")
    sn_prof, sn_cts = mean_profiles("GSE131882")

    genes = sc_prof.index.intersection(sn_prof.index)
    sc_prof, sn_prof = sc_prof.loc[genes], sn_prof.loc[genes]

    def hv(df):
        v = df.var(axis=1)
        return set(v.sort_values(ascending=False).head(N_HVG).index)
    feat = sorted(hv(sc_prof) | hv(sn_prof))
    sc_prof, sn_prof = sc_prof.loc[feat], sn_prof.loc[feat]

    ref = np.sort(np.concatenate(
        [sc_prof.values.ravel(), sn_prof.values.ravel()]))
    ref = np.interp(np.linspace(0, 1, len(feat)),
                    np.linspace(0, 1, len(ref)), ref)

    def qnorm(df):
        out = df.copy()
        for c in df.columns:
            order = df[c].rank(method="first").astype(int) - 1
            out[c] = ref[order.values]
        return out
    sc_q, sn_q = qnorm(sc_prof), qnorm(sn_prof)

    cols = {}
    for ct in sc_cts:
        cols[ct] = sc_q[ct]
    for ct in FROM_SN:
        if ct in sn_q.columns:
            cols[ct] = sn_q[ct]
    sig = pd.DataFrame(cols)
    sig = sig / (np.linalg.norm(sig.values, axis=0, keepdims=True) + 1e-9)
    return sig


def deconvolve(sig: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp_name in ("GLOM", "TUB"):
        comp = D.load_compartment(comp_name)
        for cid, cd in comp.cohorts.items():
            genes = [g for g in sig.index if g in cd.expr.index]
            B = sig.loc[genes].values
            Y = cd.expr.loc[genes].values
            for j, s in enumerate(cd.samples):
                w, _ = nnls(B, Y[:, j])
                w = w / (w.sum() + 1e-9)
                rows.append(dict(compartment=comp_name, cohort=cid, sample=s,
                                 y=int(cd.y[j]), **dict(zip(sig.columns, w))))
    return pd.DataFrame(rows)


def adjust(prop: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = [c for c in prop.columns
             if c not in ("compartment", "cohort", "sample", "y")]
    for comp_name, sub in prop.groupby("compartment"):
        p = pred[(pred.compartment == comp_name) &
                 (pred.model == "scPair_LASSO")]
        p = p.assign(score_rank=p.groupby("held_out")["p"].rank(pct=True))
        m = sub.merge(p[["sample", "score_rank"]], on="sample", how="inner")
        if len(m) < 20:
            continue
        var = m[[c for c in cells if m[c].std() > 1e-6]].var().sort_values(
            ascending=False)
        use = list(var.head(6).index)
        for label, X in (("score_only", m[["score_rank"]]),
                         ("score_plus_composition", m[["score_rank"] + use])):
            Xd = sm.add_constant(X.astype(float))
            try:
                res = sm.Logit(m.y.values, Xd).fit(disp=0, maxiter=200)
                rows.append(dict(compartment=comp_name, model=label, n=len(m),
                                 n_cell_types=len(use) if "composition" in label
                                 else 0,
                                 score_coef=round(float(
                                     res.params["score_rank"]), 3),
                                 score_p=float(res.pvalues["score_rank"]),
                                 pseudo_r2=round(float(res.prsquared), 3)))
            except Exception as exc:                        # noqa: BLE001
                rows.append(dict(compartment=comp_name, model=label, n=len(m),
                                 score_coef=np.nan, score_p=np.nan,
                                 pseudo_r2=np.nan, note=str(exc)[:50]))
    return pd.DataFrame(rows)


def main() -> None:
    sig = build_signature()
    print(f"[signature] {sig.shape[0]} genes x {sig.shape[1]} cell types: "
          f"{list(sig.columns)}")
    prop = deconvolve(sig)
    prop.to_csv(TAB / "T24b_deconvolution_complete.tsv.gz", sep="\t",
                index=False)

    cells = [c for c in prop.columns
             if c not in ("compartment", "cohort", "sample", "y")]
    print("\nmean proportion per cell type (DKD vs Control):")
    for comp_name, sub in prop.groupby("compartment"):
        mean = sub.groupby("y")[cells].mean()
        big = (mean.loc[1] - mean.loc[0]).abs().sort_values(ascending=False)
        print(f"  {comp_name}: largest change "
              f"{ {k: round(float(mean.loc[1,k]-mean.loc[0,k]),3) for k in big.head(4).index} }")

    pred = pd.read_csv(MET / "loco_predictions.tsv.gz", sep="\t")
    adj = adjust(prop, pred)
    adj.to_csv(TAB / "T25b_composition_adjustment_complete.tsv", sep="\t",
               index=False)
    print("\n===== composition-adjusted regression (full basis) =====")
    print(adj.to_string(index=False))


if __name__ == "__main__":
    main()
